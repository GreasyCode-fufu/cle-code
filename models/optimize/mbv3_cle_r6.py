"""R6 轮: MobileNetV3-Large 轻量 CLE 网络 (2026-11-08)

基座: models/MobileNetV3.py 的 mobilenet_v3_large(torchvision 同构, 键名逐一一致), features 原封不动,
仅替换分类头 —— 本协议下 CLASSIFIER_ROOTS 本就丢弃 classifier.*(960→1280 隐层 FC 冷启动),
因此任何头部重设计都不损失预训练知识, 主干严格加载(unexpected=0)不变。

创新1 候选 —— 跨尺度星乘证据头 (Cross-scale Star Evidence head, CSE)
  动机(P0 实测, experiments/r6_p0_analysis.json):
    ① 头部瓶颈: 同一微调 f16 特征上, 正则化线性头比网络自身 1.23M 参数冷启动 MLP 头
       test F1 +1.3 / 恶性敏感度 +4.0(3 seeds 配对全正), B-vs-M AUC 持平 —— 头过拟合、决策面差;
    ② /16 证据: ImageNet 预训练 f12(/16, 112ch) 的 B-vs-M AUC 0.997 为全部探针最高,
       但它不直接进入分类头(只经 /32 间接传递), 微调后降至 0.988。
  结构: u = LN(W_u·GAP(f16)) ∈ R^d,  v = LN(W_v·GAP(f12)) ∈ R^d,
        z = [u, ReLU6(u) ⊙ v],  logits = W_o·Dropout(z)
  数学: (ReLU6(u) ⊙ v)_k = ReLU6(w_u,k·p16)·(w_v,k·p12) —— 每维是 p16 与 p12 的秩-1 双线性形式,
        d 维合起来即低秩分解的跨尺度双线性池化(Hadamard 低秩双线性, Kim et al. ICLR 2017),
        以 O(d) 代价获得跨尺度二阶交互, 取代 1280 宽隐层"以宽度换容量"(参数 1.23M → 0.28M)。
  参照: 模块库 9 StarConv(StarNet "Rewrite the Stars", CVPR 2024)的 act(x1)*x2 星运算,
        由块内同尺度特征改为分类头的跨尺度证据交互。
  对照行(消融归因): lin = 线性头; mlp256 = 同宽缩头 MLP(加性); star1 = 同尺度星乘(仅 f16)。

创新2 候选 —— /16 结构感知局部聚合 (接在 features[12] 之后, 112ch@14²)
  LSAgg: LSConv(LSNet "See Large, Focus Small", CVPR 2025; 模块库 122)官方分组共享核形式 ——
         大核感知 LKP(1×1→7×7dw→1×1→1×1) 为每个位置生成 3×3 动态核(每 8 通道共享一核),
         小核聚合 SKA 按位置加权邻域: 规则腺体区核趋同、紊乱区核分化 = 显式局部异质性传感。
         (库版把核改为逐通道, 参数 ×8, 不采用; SKA 用 9 次平移乘加实现, 确定性且可导出。)
  IDCAgg: InceptionNeXt IDC(CVPR 2024; 模块库 16)大感受野分解 dw(3×3 / 1×11 / 11×1)+1×1, 大感受野对照。
  接入: y = x + BN_{γ=0}(Agg(x)) —— 零初始, 初始与 star2 逐位等价; pos="bb" 进主干(喂 /32 与头),
        pos="side" 只进头部 /16 证据通路(主干不变)。新增模块在独立 RNG 中构建, 同 seed 下与无该模块的
        模型共享完全相同的数据顺序/增广/Dropout 掩码 —— 配对比较只差模块本身。

创新3 候选 —— 训练期先验(推理零成本, 推理时不执行)
  anchor: /16 预训练证据锚定 —— 冻结的 ImageNet 预训练 features[0..12] 副本作参照, 约束微调后 f12
          的相对漂移 ||f12 − f12_pre||² / ||f12_pre||²。动机: P0 中 f12 的 B-vs-M AUC 微调后 0.997→0.988
          (CE 微调扭曲了最具判别力的中层证据; cf. L2-SP ICML'18 / LP-FT ICLR'22 的特征扭曲论)。
  edge:   MatchED(CVPR 2026; 模块库 175)式紧凑边缘头, 在 /16 回归 FOV 内 Sobel 边缘密度(逐图 z-score),
          迫使中层特征保留腺体边界几何。(soft-chamfer 面向全分辨率细边, 14² 上改为密度回归。)
  两者经 train_cle_fair 既有辅助损失钩子接入: loss += model.mfm_lambda * model.mfm.loss。
"""
import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MobileNetV3 import mobilenet_v3_large

TAP_CH = {"f6": 40, "f12": 112, "f16": 960}     # features[6]: /8; features[12]: /16; features[16]: /32
_IMNET_MEAN = (0.485, 0.456, 0.406)               # mbv3 run_meta: norm_source=imagenet
_IMNET_STD = (0.229, 0.224, 0.225)
_SOBEL = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8


def _init_head(m):
    """与 MobileNetV3 原头一致的初始化: Linear ~ N(0, 0.01), bias 0; LN 1/0。"""
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, 0, 0.01)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.LayerNorm):
            nn.init.ones_(mod.weight)
            nn.init.zeros_(mod.bias)


def _isolated(fn, seed=0):
    """在独立 RNG 中构建新增模块, 不扰动全局随机流。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return fn()


class StarEvidenceHead(nn.Module):
    """z = [u, ReLU6(u) ⊙ v_s (s ∈ sources)] -> Dropout -> Linear.
    sources=("f12",) 为跨尺度星乘; sources=("f16",) 为同尺度星乘对照(StarNet 原式)。"""

    def __init__(self, num_classes, sources=("f12",), d=256, dropout=0.2, ch=None):
        super().__init__()
        self.sources = tuple(sources)
        cin = dict(TAP_CH, **(ch or {}))
        self.u = nn.Sequential(nn.Linear(TAP_CH["f16"], d, bias=False), nn.LayerNorm(d))
        self.v = nn.ModuleDict({s: nn.Sequential(nn.Linear(cin[s], d, bias=False), nn.LayerNorm(d))
                                for s in self.sources})
        self.act = nn.ReLU6()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d * (1 + len(self.sources)), num_classes)
        _init_head(self)

    def forward(self, pooled):
        u = self.u(pooled["f16"])
        a = self.act(u)
        z = [u] + [a * self.v[s](pooled[s]) for s in self.sources]
        return self.fc(self.drop(torch.cat(z, dim=1)))


class LSAgg(nn.Module):
    """LSConv 官方分组共享核形式 + 零初始残差: y = x + BN_{γ=0}(SKA(x, LKP(x)))."""

    def __init__(self, dim, lks=7, sks=3, share=8):
        super().__init__()
        assert dim % share == 0
        h = dim // 2
        self.sks, self.ng = sks, dim // share
        self.lkp = nn.Sequential(
            nn.Conv2d(dim, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU(inplace=True),
            nn.Conv2d(h, h, lks, padding=lks // 2, groups=h, bias=False), nn.BatchNorm2d(h),
            nn.Conv2d(h, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU(inplace=True),
            nn.Conv2d(h, sks * sks * self.ng, 1),
            nn.GroupNorm(self.ng, sks * sks * self.ng))
        self.bn = nn.BatchNorm2d(dim)
        nn.init.zeros_(self.bn.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        s = self.sks
        w = self.lkp(x).view(B, self.ng, 1, s * s, H, W)
        xp = F.pad(x, (s // 2,) * 4).view(B, self.ng, C // self.ng, H + s - 1, W + s - 1)
        out = 0
        for k in range(s * s):                  # SKA: 9 次平移乘加(确定性, 可导出, 等价 unfold 实现)
            i, j = divmod(k, s)
            out = out + xp[..., i:i + H, j:j + W] * w[:, :, :, k]
        return x + self.bn(out.reshape(B, C, H, W))


class IDCAgg(nn.Module):
    """IDC 大感受野分解 dw(通道三分: 3×3 / 1×k / k×1)+ 1×1 混合 + 零初始残差。"""

    def __init__(self, dim, sq=3, band=11):
        super().__init__()
        g = dim // 3
        self.split = (dim - 2 * g, g, g)
        a = self.split[0]
        self.dw_hw = nn.Conv2d(a, a, sq, padding=sq // 2, groups=a, bias=False)
        self.dw_w = nn.Conv2d(g, g, (1, band), padding=(0, band // 2), groups=g, bias=False)
        self.dw_h = nn.Conv2d(g, g, (band, 1), padding=(band // 2, 0), groups=g, bias=False)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        nn.init.zeros_(self.bn.weight)

    def forward(self, x):
        a, b, c = torch.split(x, self.split, dim=1)
        return x + self.bn(self.pw(torch.cat([self.dw_hw(a), self.dw_w(b), self.dw_h(c)], dim=1)))


class EdgeHead(nn.Module):
    """MatchED 式紧凑边缘头(1×1 入口 + 2×[3×3-BN-ReLU] + 1×1 出口), 输出 /16 边缘密度图。"""

    def __init__(self, dim, hidden=32, blocks=2):
        super().__init__()
        layers = [nn.Conv2d(dim, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
        for _ in range(blocks):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1, bias=False), nn.BatchNorm2d(hidden),
                       nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(hidden, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def edge_density_target(x, out_hw):
    """输入(ImageNet 归一化) -> FOV 内 Sobel 边缘密度 @out_hw, 逐图 z-score; 返回 (z, keep)。
    FOV: 15×15 均值 > 0.03 后腐蚀 7px, 去掉圆形视野边界的人工强边(与 r6_p0_analysis 一致)。"""
    mean = torch.tensor(_IMNET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMNET_STD, device=x.device).view(1, 3, 1, 1)
    g = (x.float() * std + mean).mean(1, keepdim=True)
    fov = (F.avg_pool2d(g, 15, 1, 7, count_include_pad=False) > 0.03).float()
    fov = 1 - F.max_pool2d(1 - fov, 15, 1, 7)
    k = torch.stack([_SOBEL, _SOBEL.t()]).unsqueeze(1).to(x.device)
    mag = F.conv2d(g, k, padding=1).pow(2).sum(1, keepdim=True).sqrt() * fov
    s = g.shape[-1] // out_hw
    frac = F.avg_pool2d(fov, s)
    e = F.avg_pool2d(mag, s) / frac.clamp_min(1e-6)
    keep = frac > 0.95
    n = keep.sum((2, 3), keepdim=True).clamp_min(1)
    mu = (e * keep).sum((2, 3), keepdim=True) / n
    sd = ((((e - mu) * keep) ** 2).sum((2, 3), keepdim=True) / n).sqrt()
    return (e - mu) / sd.clamp_min(1e-6), keep


class MBV3R6(nn.Module):
    """features 与基座逐键一致; 头部按 head 选择; 可选 /16 聚合(r6agg)、训练期锚定(anchor_net)/边缘头(r6edge)。
    v_src: 头部 /16 证据来源 —— "out" = features[12] 线性瓶颈输出(112ch);
           "exp" = features[12] 块内扩展空间(SE 之后、投影之前, 672ch, 已过 h-swish 激活)。
    freeze_upto: 冻结 features[0..k](参数不训练 + BN 保持推理态 = 纯 ImageNet 预训练证据), -1 不冻结。"""

    def __init__(self, num_classes, head="star", sources=("f12",), d=256,
                 agg=None, agg_pos="bb", anchor_lambda=0.0, edge_lambda=0.0, v_src="out", freeze_upto=-1):
        super().__init__()
        base = mobilenet_v3_large(num_classes=num_classes)
        self.features = base.features
        self.v_src = v_src
        if head == "mlp":                       # 原头(仅用于一致性自检, 等价 mobilenet_v3_large)
            self.classifier = base.classifier
        elif head == "lin":
            self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(TAP_CH["f16"], num_classes))
        elif head == "mlp256":
            self.classifier = nn.Sequential(nn.Linear(TAP_CH["f16"], d), nn.Hardswish(inplace=True),
                                            nn.Dropout(0.2), nn.Linear(d, num_classes))
        elif head == "star":
            self.classifier = StarEvidenceHead(num_classes, sources=sources, d=d,
                                               ch={"f12": 672} if v_src == "exp" else None)
        else:
            raise ValueError(head)
        if head in ("lin", "mlp256"):           # StarEvidenceHead 已在自身 __init__ 中初始化
            _init_head(self.classifier)
        self.taps = set(sources) if head == "star" else set()

        self.agg, self.agg_pos = agg, agg_pos
        if agg == "ls":
            self.r6agg = _isolated(lambda: LSAgg(TAP_CH["f12"]))
        elif agg == "idc":
            self.r6agg = _isolated(lambda: IDCAgg(TAP_CH["f12"]))
        elif agg is not None:
            raise ValueError(agg)

        self.anchor_lambda, self.edge_lambda = anchor_lambda, edge_lambda
        if anchor_lambda > 0:                   # 权重由 checkpoint 变换从预训练 features.0-12 注入
            self.anchor_net = copy.deepcopy(self.features[:13])
            self.anchor_net.requires_grad_(False)
            for mod in self.anchor_net.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    # 与学生同用 batch 统计量归一化(训练态), 但 momentum=0 永不更新 running 统计量:
                    # 初始时学生≡参照(锚定损失恰为 0), 之后损失只反映权重漂移, 不混入 BN 训练/推理态差异
                    mod.momentum = 0.0
        if edge_lambda > 0:
            self.r6edge = _isolated(lambda: EdgeHead(TAP_CH["f12"]), seed=1)
        self.mfm_lambda = 1.0
        self._aux = None
        self.freeze_upto = freeze_upto
        if freeze_upto >= 0:
            self.features[:freeze_upto + 1].requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_upto >= 0:
            self.features[:self.freeze_upto + 1].eval()     # 冻结段 BN 保持 ImageNet running 统计量
        return self

    @property
    def mfm(self):
        """复用 train_cle_fair 既有辅助损失钩子: loss += model.mfm_lambda * model.mfm.loss。"""
        if self.anchor_lambda > 0 or self.edge_lambda > 0:
            return SimpleNamespace(loss=self._aux)
        return None

    def _aux_loss(self, x0, f12):
        loss = f12.new_zeros(()).float()
        if self.anchor_lambda > 0:
            with torch.no_grad():
                t = self.anchor_net(x0).float()
            loss = loss + self.anchor_lambda * (f12.float() - t).pow(2).mean() / t.pow(2).mean().clamp_min(1e-6)
        if self.edge_lambda > 0:
            z, keep = edge_density_target(x0, f12.shape[-1])
            pred = self.r6edge(f12).float()
            loss = loss + self.edge_lambda * F.smooth_l1_loss(pred[keep], z[keep])
        return loss

    def forward_pooled(self, x):
        x0, pooled, f12_pre, xe = x, {}, None, None
        for i, blk in enumerate(self.features):
            if i == 12 and self.v_src == "exp":     # 逐步展开 InvertedResidual, 取扩展空间(672ch)作 /16 证据
                xe = blk.block[2](blk.block[1](blk.block[0](x)))
                y = blk.block[3](xe)
                x = y + x if blk.use_res_connect else y
            else:
                x = blk(x)
            if i == 6 and "f6" in self.taps:
                pooled["f6"] = x.mean(dim=(2, 3))
            elif i == 12:
                f12_pre, e = x, x
                if self.agg is not None:
                    e = self.r6agg(x)
                    if self.agg_pos == "bb":
                        x = e
                if "f12" in self.taps:
                    pooled["f12"] = (xe if xe is not None else e).mean(dim=(2, 3))
        pooled["f16"] = x.mean(dim=(2, 3))      # = AdaptiveAvgPool2d(1) + flatten
        if self.training and (self.anchor_lambda > 0 or self.edge_lambda > 0):
            self._aux = self._aux_loss(x0, f12_pre)
        return pooled

    def forward(self, x):
        pooled = self.forward_pooled(x)
        if isinstance(self.classifier, StarEvidenceHead):
            return self.classifier(pooled)
        return self.classifier(pooled["f16"])


# ── 创新1 候选与对照 ──
def mbv3_r6_lin(num_classes=3):
    return MBV3R6(num_classes, head="lin")


def mbv3_r6_mlp256(num_classes=3):
    return MBV3R6(num_classes, head="mlp256")


def mbv3_r6_star1(num_classes=3):
    return MBV3R6(num_classes, head="star", sources=("f16",))


def mbv3_r6_star2(num_classes=3):
    return MBV3R6(num_classes, head="star", sources=("f12",))


def mbv3_r6_star3(num_classes=3):
    return MBV3R6(num_classes, head="star", sources=("f12", "f6"))


# ── 创新2 / 创新3 候选(叠加在 star2 头之上; 头部定稿后按需改基) ──
def mbv3_r6_s2_lsbb(num_classes=3):
    return MBV3R6(num_classes, head="star", sources=("f12",), agg="ls", agg_pos="bb")


def mbv3_r6_s2_lsside(num_classes=3):
    return MBV3R6(num_classes, head="star", sources=("f12",), agg="ls", agg_pos="side")


def mbv3_r6_s2_idcbb(num_classes=3):
    return MBV3R6(num_classes, head="star", sources=("f12",), agg="idc", agg_pos="bb")


def mbv3_r6_s2_anc1(num_classes=3):
    return MBV3R6(num_classes, head="star", sources=("f12",), anchor_lambda=1.0)


def mbv3_r6_s2_anc03(num_classes=3):
    return MBV3R6(num_classes, head="star", sources=("f12",), anchor_lambda=0.3)


def mbv3_r6_s2_edge(num_classes=3):
    return MBV3R6(num_classes, head="star", sources=("f12",), edge_lambda=0.2)


# ── MB 批后的修订候选(MB 六候选全负; 见 R6_REPORT.md §3) ──
# 创新1 修订: /16 证据取块内扩展空间(线性瓶颈理论: 112ch 瓶颈输出是压缩流形, 672ch 激活扩展空间信息更全)
def mbv3_r6_star2x(num_classes=3):
    return MBV3R6(num_classes, head="star", sources=("f12",), v_src="exp")


# 训练期证据保持(推理零成本): 冻结预训练前段(参数不训 + BN 推理态), 动机 P0-F2(微调使 /16 AUC_BM 0.997→0.988)
def mbv3_r6_s2_fz8(num_classes=3):
    return MBV3R6(num_classes, head="star", sources=("f12",), freeze_upto=6)


def mbv3_r6_s2_fz16(num_classes=3):
    return MBV3R6(num_classes, head="star", sources=("f12",), freeze_upto=12)


def mbv3_r6_s2x_fz16(num_classes=3):
    return MBV3R6(num_classes, head="star", sources=("f12",), v_src="exp", freeze_upto=12)


# ── 解耦凸证据头(Decoupled Convex Evidence head, DCE) —— P0d/P0e 离线验证后的主线 ──
# 阶段 1: 主干按原协议端到端微调(与 baseline 逐位一致); 阶段 2(experiments/r6_refit.py): 冻结主干,
# 在多尺度 GAP 证据上以强 L2 多项逻辑回归拟合分类头, 正则强度仅在 val 上按 Macro-F1 选择(不触碰 test)。
# 推理结构: z = [GAP(f16), GAP(f12), GAP(f6), GAP(f3)] -> (z-μ)/σ -> Linear(D->K); μ/σ 为 buffer(可折叠进 Linear)。
CVX_TAP_IDX = {"f3": 3, "f6": 6, "f12": 12, "f16": 16}
CVX_TAP_CH = {"f3": 24, "f6": 40, "f12": 112, "f16": 960}


class ConvexEvidenceHead(nn.Module):
    def __init__(self, num_classes, taps=("f16",)):
        super().__init__()
        self.taps = tuple(taps)
        d = sum(CVX_TAP_CH[t] for t in self.taps)
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sigma", torch.ones(d))
        self.fc = nn.Linear(d, num_classes)

    def forward(self, pooled):
        z = torch.cat([pooled[t] for t in self.taps], dim=1)
        return self.fc((z - self.mu) / self.sigma)


def fov_weights(x):
    """FOV 感知池化权重: 输入(ImageNet 归一化)反归一化灰度 -> 4× 平均下采样到 1/4 分辨率(=f3 尺度) -> 3×3 平滑
    (约等于原分辨率 12px 窗口, 填平腺腔等小暗区) -> > 0.03 即圆形视野内组织区(视野外近黑背景占 25–31%, P0)。
    在 1/4 分辨率上计算: 掩码只需在各抽头尺度(56²/28²/14²/7²)上准确, 全分辨率 15×15 池化会使 CPU 单线程延迟 +43%。
    返回 (B,1,H/4,W/4) 掩码, 由各尺度自适应平均后归一化为池化权重; 无暗背景的图像退化为普通 GAP。"""
    mean = torch.tensor(_IMNET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMNET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    g = F.avg_pool2d((x * std + mean).mean(1, keepdim=True), 4)
    return (F.avg_pool2d(g, 3, 1, 1, count_include_pad=False) > 0.03).to(x.dtype)


class MBV3Convex(nn.Module):
    """mbv3 主干(键名与基座一致) + 解耦凸证据头。pool="gap" 普通全局平均; "mgap" FOV 感知掩码平均。"""

    def __init__(self, num_classes, taps=("f16",), pool="gap"):
        super().__init__()
        self.features = mobilenet_v3_large(num_classes=num_classes).features
        self.classifier = ConvexEvidenceHead(num_classes, taps)
        self.tap_idx = {CVX_TAP_IDX[t]: t for t in taps}
        self.pool = pool

    def forward_pooled(self, x):
        pooled = {}
        fov = fov_weights(x) if self.pool == "mgap" else None
        for i, blk in enumerate(self.features):
            x = blk(x)
            if i in self.tap_idx:
                if fov is None:
                    pooled[self.tap_idx[i]] = x.mean(dim=(2, 3))
                else:
                    w = F.adaptive_avg_pool2d(fov, x.shape[-2:])
                    w = w / w.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                    pooled[self.tap_idx[i]] = (x * w).sum(dim=(2, 3))
        return pooled

    def forward(self, x):
        return self.classifier(self.forward_pooled(x))


def mbv3_dce16(num_classes=3):
    return MBV3Convex(num_classes, taps=("f16",))


def mbv3_dce2s(num_classes=3):
    return MBV3Convex(num_classes, taps=("f16", "f12"))


def mbv3_dce4s(num_classes=3):
    return MBV3Convex(num_classes, taps=("f16", "f12", "f6", "f3"))


def mbv3_dce4m(num_classes=3):
    return MBV3Convex(num_classes, taps=("f16", "f12", "f6", "f3"), pool="mgap")


# 证据保持(阶段 1 冻结 /2–/16, 原 MLP 头) —— 与 baseline 同 seed 逐位配对(仅差冻结)
def mbv3_r6_fz16(num_classes=3):
    return MBV3R6(num_classes, head="mlp", freeze_upto=12)
