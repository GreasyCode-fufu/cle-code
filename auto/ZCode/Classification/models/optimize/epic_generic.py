"""R6 实验 2: EPIC 三组件的跨主干通用实现(ConvNeXt-T / StarNet-S2 / FasterNet-T0)。

与 MobileNetV3 版(models/optimize/mbv3_cle_r6.py)同构:
  ① 证据保持冻结: 阶段 1 冻结前三个 stage(/4–/16)的参数, 冻结段 BN 保持推理态(ConvNeXt 为 LayerNorm, 无统计量),
     只微调最后一个 stage(/32)与原头部 —— 工厂 <arch>_fz3, 与 baseline 同 seed 初始逐位一致, 仅差冻结。
  ② 解耦凸证据头: 阶段 2(experiments/r6_refit.py)冻结主干, 强 L2 多项逻辑回归拟合, C 仅在 val 上选。
  ③ FOV 感知多尺度证据: 四个 stage 输出(/4, /8, /16, /32)在圆形视野组织区做掩码平均池化(fov_weights 同 mbv3 版)。
阶段 2 输出结构 EPICGeneric = 原主干(去掉原头部/最终归一化)+ 证据池化 + (z−μ)/σ + Linear。
不同阶段 1 来源使用不同注册名(输出目录按注册名区分), 见文件末尾 REGISTRY_NAMES。
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.optimize.mbv3_cle_r6 import fov_weights


def _build(arch, num_classes):
    if arch == "convnext_tiny":
        from models.Convnext import convnext_tiny
        return convnext_tiny(num_classes=num_classes)
    if arch == "starnet_s2":
        import timm
        return timm.create_model("starnet_s2", pretrained=False, num_classes=num_classes)
    if arch == "fast_t0":
        from models.FasterNet import fast_t0
        return fast_t0(num_classes=num_classes)
    raise ValueError(arch)


# 每个主干: 前三 stage 模块(冻结用)、四个证据抽头的通道数、原头部键前缀(阶段 2 丢弃)
SPECS = {
    "convnext_tiny": {"dims": (96, 192, 384, 768), "head_prefixes": ("norm.", "head.")},
    "starnet_s2": {"dims": (32, 64, 128, 256), "head_prefixes": ("norm.", "head.")},
    "fast_t0": {"dims": (40, 80, 160, 320), "head_prefixes": ("avgpool_pre_head.", "head.")},
}
TAPS = ("s1", "s2", "s3", "s4")


def _front_modules(arch, m):
    if arch == "convnext_tiny":
        return [m.downsample_layers[i] for i in range(3)] + [m.stages[i] for i in range(3)]
    if arch == "starnet_s2":
        return [m.stem] + [m.stages[i] for i in range(3)]
    if arch == "fast_t0":
        return [m.patch_embed] + [m.stages[i] for i in range(5)]   # stages[0..4] = BasicStage/Merge 直到 /16
    raise ValueError(arch)


def _stage_outputs(arch, m, x):
    outs = []
    if arch == "convnext_tiny":
        for i in range(4):
            x = m.stages[i](m.downsample_layers[i](x))
            outs.append(x)
    elif arch == "starnet_s2":
        x = m.stem(x)
        for s in m.stages:
            x = s(x)
            outs.append(x)
    elif arch == "fast_t0":
        x = m.patch_embed(x)
        for i, s in enumerate(m.stages):
            x = s(x)
            if i in (0, 2, 4, 6):
                outs.append(x)
    return outs


def _bn_always_eval(bn):
    def train(self, mode=True):
        return nn.Module.train(self, False)
    bn.train = types.MethodType(train, bn)


def build_fz3(arch, num_classes):
    """阶段 1(①): 原主干与原头部, 冻结前三 stage 参数, 冻结段 BN 恒为推理态。"""
    m = _build(arch, num_classes)
    for mod in _front_modules(arch, m):
        mod.requires_grad_(False)
        for sub in mod.modules():
            if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                _bn_always_eval(sub)
    return m


class GenericConvexHead(nn.Module):
    def __init__(self, num_classes, taps, dims):
        super().__init__()
        self.taps = tuple(taps)
        d = sum(dims[t] for t in self.taps)
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sigma", torch.ones(d))
        self.fc = nn.Linear(d, num_classes)

    def forward(self, pooled):
        z = torch.cat([pooled[t] for t in self.taps], dim=1)
        return self.fc((z - self.mu) / self.sigma)


class EPICGeneric(nn.Module):
    """阶段 2 部署结构: backbone(原主干, 原头部置为 Identity) + 证据池化 + 凸证据头。"""

    def __init__(self, arch, num_classes, taps=TAPS, pool="mgap"):
        super().__init__()
        self.arch, self.pool = arch, pool
        self.backbone = _build(arch, num_classes)
        for p in SPECS[arch]["head_prefixes"]:
            setattr(self.backbone, p.rstrip("."), nn.Identity())
        dims = dict(zip(TAPS, SPECS[arch]["dims"]))
        self.classifier = GenericConvexHead(num_classes, taps, dims)

    def load_stage1(self, sd):
        """阶段 1 checkpoint(原主干键名)-> backbone.*; 丢弃原头部键。"""
        heads = SPECS[self.arch]["head_prefixes"]
        mapped = {"backbone." + k: v for k, v in sd.items() if not k.startswith(heads)}
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        assert not unexpected and all(k.startswith("classifier.") for k in missing), (missing, unexpected)

    def forward_pooled(self, x):
        outs = _stage_outputs(self.arch, self.backbone, x)
        fov = fov_weights(x) if self.pool == "mgap" else None
        pooled = {}
        for t, o in zip(TAPS, outs):
            if t not in self.classifier.taps:
                continue
            if fov is None:
                pooled[t] = o.mean(dim=(2, 3))
            else:
                w = F.adaptive_avg_pool2d(fov, o.shape[-2:])
                w = w / w.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                pooled[t] = (o * w).sum(dim=(2, 3))
        return pooled

    def forward(self, x):
        return self.classifier(self.forward_pooled(x))


# ── 注册名(阶段 1 两种 + 阶段 2 六种来源/配置) ──
#   <a>_fz3                 阶段 1: ① 冻结前三 stage(原头部)
#   <a>_fz3_dce_last        ①+②   (fz3 主干, 末 stage GAP)
#   <a>_fz3_dce4m           ①+②+③(fz3 主干, 四抽头 FOV 掩码) = EPIC
#   <a>_dce_last / <a>_dce4m   ② / ②+③(全量微调 baseline 主干)
#   <a>_lp_dce4s / <a>_lp_dce4m 纯 ImageNet 主干: 门控 LP_val(四抽头 GAP) / 全保留档
_CFG = {"fz3_dce_last": (("s4",), "gap"), "fz3_dce4m": (TAPS, "mgap"), "dce_last": (("s4",), "gap"),
        "dce4m": (TAPS, "mgap"), "lp_dce4s": (TAPS, "gap"), "lp_dce4m": (TAPS, "mgap")}
REGISTRY_NAMES = []
for _a in SPECS:
    def _mk_fz(a=_a):
        return lambda num_classes=3: build_fz3(a, num_classes)
    globals()[f"{_a}_fz3"] = _mk_fz()
    REGISTRY_NAMES.append(f"{_a}_fz3")
    for _k, (_taps, _pool) in _CFG.items():
        def _mk(a=_a, taps=_taps, pool=_pool):
            return lambda num_classes=3: EPICGeneric(a, num_classes, taps, pool)
        globals()[f"{_a}_{_k}"] = _mk()
        REGISTRY_NAMES.append(f"{_a}_{_k}")
