"""库模块量产适配器 —— 2026-10-24

目的: 把 shared/module_library(425个第三方模块)中的结构快速包成"零初始化门控
残差"变体批量实验(用户指令: 大量尝试, 不让GPU休息)。

GatedWrap: 通用包装器  x + gamma(0初始化) * BN(1x1( block(1x1(x)) ))
block 只需实现 r->r(r=ch/4) 的同分辨率映射。

本文件首批 3 个适配(刻意避开已证伪的通道注意力/频域乘性注意力家族):
- resnet34_lib_haar:  Haar 小波高低频分解(246/217 灵感) —— 频带结构, 但在门控内
- resnet34_lib_msctx: 方向池化上下文(222/240 灵感) —— H/W 条带均值上下文
- resnet34_lib_lcattn: 线性注意力全局上下文(218 灵感) —— elu+1 核, 免softmax
插入位置 layer3/4 出口。注册名 lib_ 前缀, 白名单 {"lib"}。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedWrap(nn.Module):
    """库模块 -> 零初始化门控残差的通用包装。"""

    def __init__(self, ch, block_fn, r_div=4):
        super().__init__()
        r = max(ch // r_div, 16)
        self.reduce = nn.Sequential(nn.Conv2d(ch, r, 1, bias=False),
                                    nn.BatchNorm2d(r), nn.ReLU(inplace=True))
        self.block = block_fn(r)
        self.fuse = nn.Conv2d(r, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.gamma = nn.Parameter(torch.zeros(ch))

    def forward(self, x):
        g = self.gamma.view(1, -1, 1, 1)
        return x + g * self.bn(self.fuse(self.block(self.reduce(x))))


def haar_block(r):
    """Haar 高低频: avgpool 下采样为低频, 残差为高频, 拼接后 1x1 融合。"""
    class B(nn.Module):
        def __init__(self):
            super().__init__()
            self.fuse = nn.Conv2d(r * 2, r, 1, bias=False)

        def forward(self, x):
            h, w = x.shape[-2:]
            pad = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")  # 奇数尺寸补偶
            low = F.interpolate(F.avg_pool2d(pad, 2), size=(h, w), mode="nearest")
            return self.fuse(torch.cat([low, x - low], dim=1))
    return B()


def msctx_block(r):
    """方向池化上下文: H/W 两个方向的均值图广播拼接融合。"""
    class B(nn.Module):
        def __init__(self):
            super().__init__()
            self.fuse = nn.Conv2d(r * 3, r, 1, bias=False)

        def forward(self, x):
            ph = x.mean(dim=3, keepdim=True).expand_as(x)   # (r,H,1)->广播
            pw = x.mean(dim=2, keepdim=True).expand_as(x)   # (r,1,W)->广播
            return self.fuse(torch.cat([x, ph, pw], dim=1))
    return B()


def lcattn_block(r):
    """线性注意力全局上下文(elu+1 核, O(Ncd)): out_i = q_i^T(Σ k_j v_j^T)/q_i^T(Σ k_j)。"""
    class B(nn.Module):
        def __init__(self):
            super().__init__()
            self.qk = nn.Conv2d(r, r * 2, 1, bias=False)
            self.v = nn.Conv2d(r, r, 3, padding=1, groups=r, bias=False)
            self.out = nn.Conv2d(r, r, 1, bias=False)

        def forward(self, x):
            b, c, h, w = x.shape
            q, k = self.qk(x).chunk(2, dim=1)
            v = self.v(x)
            q = F.elu(q) + 1
            k = F.elu(k) + 1
            qf = q.flatten(2)                                # (b,c,N)
            kf = k.flatten(2)
            vf = v.flatten(2)
            KV = torch.einsum("bcn,bdn->bcd", kf, vf)        # (b,c,c)
            Ksum = kf.sum(dim=2)                             # (b,c)
            num = torch.einsum("bcn,bcd->bnd", qf, KV)       # (b,N,d)
            den = torch.einsum("bcn,bc->bn", qf, Ksum).unsqueeze(-1) + 1e-6
            ctx = (num / den).transpose(1, 2).reshape(b, c, h, w)
            return self.out(x + ctx)
    return B()


class _Base(nn.Module):
    """vanilla ResNet + ModuleDict('lib') 于 layer3/4 出口。"""

    def __init__(self, blocks_num, num_classes, include_top, block_fn):
        super().__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._mk(64, blocks_num[0])
        self.layer2 = self._mk(128, blocks_num[1], 2)
        self.layer3 = self._mk(256, blocks_num[2], 2)
        self.layer4 = self._mk(512, blocks_num[3], 2)
        self.lib = nn.ModuleDict({"3": GatedWrap(256, block_fn), "4": GatedWrap(512, block_fn)})
        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _mk(self, ch, n, stride=1):
        from models.optimize.resnet34_msec import BasicBlock
        ds = None
        if stride != 1 or self.in_channel != ch:
            ds = nn.Sequential(nn.Conv2d(self.in_channel, ch, 1, stride=stride, bias=False),
                               nn.BatchNorm2d(ch))
        layers = [BasicBlock(self.in_channel, ch, stride=stride, downsample=ds)]
        self.in_channel = ch
        layers += [BasicBlock(ch, ch) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.lib["3"](self.layer3(x))
        x = self.lib["4"](self.layer4(x))
        if self.include_top:
            x = self.fc(torch.flatten(self.avgpool(x), 1))
        return x


class _Hybrid(_Base):
    """layer3 出口用 DilGate(dil234), layer4 出口用 GatedWrap(msctx)。"""

    def __init__(self, blocks_num, num_classes, include_top):
        from models.optimize.resnet34_dilgate import DilGate
        super().__init__(blocks_num, num_classes, include_top, msctx_block)
        self.lib = nn.ModuleDict({
            "3": DilGate(256, rates=(2, 3, 4)),
            "4": GatedWrap(512, msctx_block),
        })


def resnet34_lib_msctx_haar(num_classes=1000, include_top=True):
    # 组合块: 方向上下文 + haar 高低频 串联
    return _Base([3, 4, 6, 3], num_classes, include_top,
                 lambda r: nn.Sequential(msctx_block(r), haar_block(r)))


def resnet34_hybrid_dil_msctx(num_classes=1000, include_top=True):
    return _Hybrid([3, 4, 6, 3], num_classes, include_top)


def resnet34_lib_haar(num_classes=1000, include_top=True):
    return _Base([3, 4, 6, 3], num_classes, include_top, haar_block)


def resnet34_lib_msctx(num_classes=1000, include_top=True):
    return _Base([3, 4, 6, 3], num_classes, include_top, msctx_block)


def resnet34_lib_lcattn(num_classes=1000, include_top=True):
    return _Base([3, 4, 6, 3], num_classes, include_top, lcattn_block)
