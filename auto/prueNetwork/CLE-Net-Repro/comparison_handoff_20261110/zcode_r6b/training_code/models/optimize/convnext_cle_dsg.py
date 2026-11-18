"""DSG-Net: ConvNeXt-T + 方向-周期-统计谱 三个零初始化条件增强创新 (2026-10-29 设计)。

针对 CLE 图像分类的三类判别证据, 每个创新都是 identity-at-init 残差支路,
完整保留 ImageNet 预训练函数与 state-dict 兼容(backbone 键名不变):

A. DOSE (Direction-Orientation Selective Enhancement) 方向门控可分离上下文块
   灵感: module_library 219 DHOGSA(AAAI26, 梯度方向作为条件) + 234 DEGConv(CVPR26,
   方向条带边缘门控) + 138 StripConv(AAAI26)。机制: 有限差分统计 4 方向梯度能量
   (水平/垂直/两对角), 归一化为逐通道方向分布, 门控 1×k 与 k×1 深度条带卷积及
   其两个共享权重复合(对角)响应; 逐通道零初始化尺度 => 初始严格等于原 Block。
   与既有工作的区别: CLE-Net SCG 为无条件固定条带; EGB 为各向同性边缘能量;
   本模块让感受野形状由内容方向结构谱决定。

B. PTG (Periodicity Texture Gate) 周期纹理自相关空间门控 (stage2/3 后)
   灵感: module_library 250 AFFN(一区TOP, 频域自相关挖掘周期结构)。机制: 8 个
   整数位移下逐位置通道余弦自相似度之和 -> tanh 压缩为周期显著性空间图, 以零
   初始化逐通道增益做乘性自门控。CLE 腺体准周期排列 vs 坏死/模糊区域的结构性
   对比, 亦适配超声散斑/胸片纹理/细粒度条纹。

C. GDS (Global Direction Spectrum) 全局方向谱描述子 (分类头前)
   与 A 同源复用方向统计: stage2/3/4 三级 4 方向能量谱+总梯度能量(15 维),
   小 MLP 经 rms 校准零初始化残差注入全局平均描述子, 补偿 GAP 丢弃的全局
   结构方向证据(区别于已失败的 evidence-pool 多尺度头)。
"""
import torch
from torch import nn
from torch.nn import functional as F

from models.Convnext import ConvNeXt, Block


def _diffs(xf):
    """fp32 有限差分: 返回 (gx, gy), 与输入同形状(边缘复制填充)。"""
    gx = F.pad(xf[:, :, 1:, :] - xf[:, :, :-1, :], (0, 0, 0, 1))
    gy = F.pad(xf[:, :, :, 1:] - xf[:, :, :, :-1], (0, 1, 0, 0))
    return gx, gy


def _orientation_energy(xf):
    """4 方向能量 (B,4,C): |gx|, |gy|, |gx+gy|, |gx-gy| 的空间均值。fp32。"""
    gx, gy = _diffs(xf)
    return torch.stack((gx.abs().mean((2, 3)), gy.abs().mean((2, 3)),
                        (gx + gy).abs().mean((2, 3)), (gx - gy).abs().mean((2, 3))), dim=1)


class _DOSECore(nn.Module):
    def __init__(self, dim, k=9):
        super().__init__()
        self.h = nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), groups=dim)
        self.v = nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), groups=dim)
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, y):
        with torch.amp.autocast(y.device.type, enabled=False):
            e = _orientation_energy(y.float())
            gate = (e / e.sum(1, keepdim=True).clamp_min(1e-6)).to(y.dtype)  # (B,4,C)
        hh = self.h(y)
        vv = self.v(y)
        d1 = self.h(vv)   # 共享权重的复合响应: 对角结构(可分基不正交方向)
        d2 = self.v(hh)
        g = gate.unsqueeze(-1).unsqueeze(-1)                               # (B,4,C,1,1)
        out = (g[:, 0] * hh + g[:, 1] * vv + g[:, 2] * d1 + g[:, 3] * d2)
        return y + self.scale[None, :, None, None] * out


class DOSEBlock(Block):
    """ConvNeXt Block + DOSE 方向门控支路(作用于 dwconv 输出)。初始恒等。"""

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6, k=9):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.dose = _DOSECore(dim, k)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.dose(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class _PTGCore(nn.Module):
    """8 位移逐位置自相似度(通道余弦)之和 -> 周期显著性图 -> 零初始化乘性自门控。"""

    SHIFTS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (2, 0), (0, 2))

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            fn = F.normalize(x.float(), dim=1, eps=1e-6)
            acc = 0.
            for dy, dx in self.SHIFTS:
                a = fn[..., max(dy, 0):fn.shape[2] + min(dy, 0),
                       max(dx, 0):fn.shape[3] + min(dx, 0)]
                b = fn[..., max(-dy, 0):fn.shape[2] + min(-dy, 0),
                       max(-dx, 0):fn.shape[3] + min(-dx, 0)]
                sim = (a * b).sum(1)                                     # (B,h,w)
                acc = acc + F.pad(sim, (max(dx, 0), max(-dx, 0),
                                        max(dy, 0), max(-dy, 0)))
            w = torch.tanh(acc / len(self.SHIFTS))                       # (B,H,W) in (-1,1)
        return x + self.scale[None, :, None, None] * x * w.to(x.dtype)[:, None]


class _OCACore(nn.Module):
    """OCA 方向相干性空间门控(v2, 取代v1加性卷积支路): 4方向局部能量的集中度
    (方向组织度, 恶性腺体紊乱 vs 正常规则排列的组织学线索; 灵感 219 HOG条件 +
    233 统计归一化加权) -> 逐位置 sigmoid 权重 -> 零初始化逐通道乘性调制。零参数核心。"""

    def __init__(self, dim, window=3):
        super().__init__()
        self.window = window
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            xf = x.float()
            gx, gy = _diffs(xf)
            k = self.window
            e = torch.stack((F.avg_pool2d(gx.abs(), k, 1, k // 2, count_include_pad=False),
                             F.avg_pool2d(gy.abs(), k, 1, k // 2, count_include_pad=False),
                             F.avg_pool2d((gx + gy).abs(), k, 1, k // 2, count_include_pad=False),
                             F.avg_pool2d((gx - gy).abs(), k, 1, k // 2, count_include_pad=False)),
                            dim=1)                                          # (B,4,C,H,W)
            dom = e.max(1).values / e.sum(1).clamp_min(1e-6)                # (B,C,H,W) 方向集中度
            w = torch.sigmoid((dom - dom.mean((2, 3), keepdim=True))
                              / (dom.std((2, 3), keepdim=True) + 1e-5))
        return x + self.scale[None, :, None, None] * x * w.to(x.dtype)


class OCAWrapBlock(Block):
    """ConvNeXt Block + 块内 OCA(作用于 dwconv 输出, EGB 同款布局)。初始恒等。"""

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6, window=3):
        super().__init__(dim, drop_rate, layer_scale_init_value)
        self.oca = _OCACore(dim, window=window)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.oca(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class _GDSCore(nn.Module):
    """stage2/3/4 三级方向谱(4 能量+1 总梯度, 每级5维) -> MLP -> rms 校准零初始化残差。"""

    def __init__(self, dim_out=768, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(15), nn.Linear(15, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim_out))
        self.scale = nn.Parameter(torch.zeros(dim_out))

    def spectrum(self, xf):
        gx, gy = _diffs(xf)
        e = torch.stack((gx.abs().mean((1, 2, 3)), gy.abs().mean((1, 2, 3)),
                         (gx + gy).abs().mean((1, 2, 3)), (gx - gy).abs().mean((1, 2, 3)),
                         (gx.abs() + gy.abs()).mean((1, 2, 3))), dim=1)   # (B,5)
        return e

    def forward(self, desc, f2, f3, f4):
        with torch.amp.autocast(desc.device.type, enabled=False):
            spec = torch.cat((self.spectrum(f2.float()), self.spectrum(f3.float()),
                              self.spectrum(f4.float())), dim=1)          # (B,15)
            correction = self.mlp(spec)
            rms = desc.detach().float().square().mean(-1, keepdim=True).clamp_min(1e-6).sqrt()
            correction = correction / correction.square().mean(-1, keepdim=True).clamp_min(1e-6).sqrt()
        return desc + self.scale[None, :] * (correction * rms).to(desc.dtype)


class DSGConvNeXt(ConvNeXt):
    def __init__(self, num_classes, dose='all', dose_k=9, ptg=(2, 3), gds=False, oca=(),
                 oca_window=3, oca_block=False):
        """dose: 'all'/'late'/None; ptg: PTG 的 stage 索引; gds: 方向谱头; oca: OCA 的 stage 索引;
        oca_block: OCA 嵌入 stages1-3 每个 Block 内部(EGB 同款布局, 优先于 oca)。"""
        super().__init__(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes)
        depths, dims = [3, 3, 9, 3], [96, 192, 384, 768]
        if oca_block:
            for i in range(3):
                for j in range(depths[i]):
                    self.stages[i][j] = OCAWrapBlock(dims[i], window=oca_window)
        elif dose == 'all':
            for i in range(3):
                for j in range(depths[i]):
                    self.stages[i][j] = DOSEBlock(dims[i], k=dose_k)
        elif dose == 'late':
            for j in (2, 5, 8):
                self.stages[2][j] = DOSEBlock(dims[2], k=dose_k)
        self.ptg_ids = tuple(ptg) if ptg else ()
        for i in self.ptg_ids:
            setattr(self, f'ptg{i}', _PTGCore(dims[i]))
        self.oca_ids = tuple(oca) if oca else ()
        for i in self.oca_ids:
            setattr(self, f'oca{i}', _OCACore(dims[i], window=oca_window))
        self.gds = _GDSCore(dims[-1]) if gds else None

    def forward_features(self, x):
        f2 = f3 = None
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.ptg_ids:
                x = getattr(self, f'ptg{i}')(x)
            if i in self.oca_ids:
                x = getattr(self, f'oca{i}')(x)
            if i == 1:
                f2 = x
            elif i == 2:
                f3 = x
        desc = self.norm(x.mean([-2, -1]))
        if self.gds is not None:
            desc = self.gds(desc, f2, f3, x)
        return desc


def convnext_tiny_cle_dose(num_classes):
    return DSGConvNeXt(num_classes, dose='all', dose_k=9, ptg=(), gds=False)


def convnext_tiny_cle_dose_late(num_classes):
    return DSGConvNeXt(num_classes, dose='late', dose_k=15, ptg=(), gds=False)


def convnext_tiny_cle_ptg(num_classes):
    return DSGConvNeXt(num_classes, dose=None, ptg=(1, 2), gds=False)


def convnext_tiny_cle_gds(num_classes):
    return DSGConvNeXt(num_classes, dose=None, ptg=(), gds=True)


def convnext_tiny_cle_dose_ptg(num_classes):
    return DSGConvNeXt(num_classes, dose='all', dose_k=9, ptg=(1, 2), gds=False)


def convnext_tiny_cle_dose_late_ptg(num_classes):
    return DSGConvNeXt(num_classes, dose='late', dose_k=15, ptg=(1, 2), gds=False)


def convnext_tiny_cle_ptg_gds(num_classes):
    return DSGConvNeXt(num_classes, dose=None, ptg=(1, 2), gds=True)


def convnext_tiny_cle_oca(num_classes):
    return DSGConvNeXt(num_classes, dose=None, ptg=(), oca=(1, 2), gds=False)


def convnext_tiny_cle_oca_gds(num_classes):
    return DSGConvNeXt(num_classes, dose=None, ptg=(), oca=(1, 2), gds=True)


def convnext_tiny_cle_oca_ptg(num_classes):
    return DSGConvNeXt(num_classes, dose=None, ptg=(1, 2), oca=(1, 2), gds=False)


def convnext_tiny_cle_oca_wide(num_classes):
    """OCA 宽窗版: stages 0-2 全覆盖 + 窗口 5。"""
    return DSGConvNeXt(num_classes, dose=None, ptg=(), oca=(0, 1, 2), oca_window=5, gds=False)


def convnext_tiny_cle_oca_block(num_classes):
    """OCA 嵌入 stages1-3 全部 Block 内部(布局类比 EGB)。"""
    return DSGConvNeXt(num_classes, dose=None, ptg=(), oca_block=True, gds=False)


def convnext_tiny_cle_oca_strip(num_classes):
    """OCA(stage2/3后门控) + 稀疏方向条带残差(stage3块2/5/8)。"""
    return DSGConvNeXt(num_classes, dose='late', dose_k=15, ptg=(), oca=(1, 2), gds=False)


def convnext_tiny_cle_oca_strip_ptg(num_classes):
    return DSGConvNeXt(num_classes, dose='late', dose_k=15, ptg=(1, 2), oca=(1, 2), gds=False)


def convnext_tiny_cle_oca_strip_gds(num_classes):
    return DSGConvNeXt(num_classes, dose='late', dose_k=15, ptg=(), oca=(1, 2), gds=True)


def convnext_tiny_cle_oca_strip_ptg_gds(num_classes):
    """R5 全栈: OCA + 方向条带(dose-late) + PTG + GDS。"""
    return DSGConvNeXt(num_classes, dose='late', dose_k=15, ptg=(1, 2), oca=(1, 2), gds=True)


def convnext_tiny_cle_oca_ptg_gds(num_classes):
    """DSG-Net v2: OCA(方向相干门控) + PTG(周期门控) + GDS(方向谱头)。"""
    return DSGConvNeXt(num_classes, dose=None, ptg=(1, 2), oca=(1, 2), gds=True)


def convnext_tiny_cle_dose_late_gds(num_classes):
    return DSGConvNeXt(num_classes, dose='late', dose_k=15, ptg=(), gds=True)


def convnext_tiny_cle_dose_ptg_gds(num_classes):
    """最终 DSG-Net: A(DOSE) + B(PTG) + C(GDS) 全叠加。"""
    return DSGConvNeXt(num_classes, dose='all', dose_k=9, ptg=(1, 2), gds=True)


def convnext_tiny_cle_dose_late_ptg_gds(num_classes):
    return DSGConvNeXt(num_classes, dose='late', dose_k=15, ptg=(1, 2), gds=True)
