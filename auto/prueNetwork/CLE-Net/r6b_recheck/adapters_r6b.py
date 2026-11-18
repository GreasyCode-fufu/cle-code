"""R6-B 复检任务: 在 CLE-Net 项目环境(torch 2.5.1/cle_topnets)中重跑 CLE-Net、ConvNeXt-T 基线与
4 个最新顶会网络(Jumbo-small/GmNet-s3 ICLR'26, iFormer-S ICLR'25, LSNet-S CVPR'25), 3 seeds(42/43/44)。

- 官方代码: /home/login/data/cle/auto/ZCode/Classification/sota_nets_2026/(只读引用, 零改动)。
- 权重: /home/login/data/cle/sunfuxiang/classifyML/pth/sota2026/*.pth(官方发布文件经受限反序列化的纯张量转存)。
- 本文件只被 r6b_recheck/run_one.py 使用, 不修改本仓库任何现有文件。
"""
import dataclasses
import importlib.util
import os
import sys
import types

import torch

SOTA = "/home/login/data/cle/auto/ZCode/Classification/sota_nets_2026"


def _load_pkg(pkg_name, rel_dir, filename):
    d = os.path.join(SOTA, rel_dir)
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [d]
        sys.modules[pkg_name] = pkg
    key = f"{pkg_name}.{filename}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(d, filename + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


# timm>=0.9 兼容补丁(LSNet 按 timm 0.5 写法, 导入期构造 PretrainedCfg(hf_hub=...)): 丢弃未知字段
import timm.models._registry as _timm_reg  # noqa: E402

if getattr(_timm_reg.PretrainedCfg, "__module__", "") != __name__:
    _PC_FIELDS = {f.name for f in dataclasses.fields(_timm_reg.PretrainedCfg)}

    class _PretrainedCfgCompat(_timm_reg.PretrainedCfg):
        def __init__(self, **kw):
            super().__init__(**{k: v for k, v in kw.items() if k in _PC_FIELDS and k != "hf_hub"})

    _timm_reg.PretrainedCfg = _PretrainedCfgCompat
    import timm.models.registry as _timm_reg2
    _timm_reg2.PretrainedCfg = _PretrainedCfgCompat


def jumbo_small(num_classes=3):
    _cfg = _load_pkg("_r6b_jumbo", "Jumbo_ICLR2026", "configs")
    _load_pkg("_r6b_jumbo", "Jumbo_ICLR2026/models", "vision_transformer")   # 相对导入依赖, 先显式加载
    _mj = _load_pkg("_r6b_jumbo", "Jumbo_ICLR2026/models", "jumbo")
    kw = _cfg.get_config("small")
    J = kw.pop("J", None)
    kw["num_classes"] = num_classes
    return _mj.Jumbo(torch.device("cpu"), J, **kw)


def gmnet_s3(num_classes=3):
    import torch.nn as nn
    g = _load_pkg("_r6b_gmnet", "GmNet_ICLR2026", "gment")
    # 按官方发布 checkpoint 的实际结构构建(f12_bn/dwconv2_bn 为 True, 其余同官方 gmnet_s3 工厂)
    return g.Model(num_classes=num_classes, embed_dim=[48, 96, 192, 384], depths=[3, 3, 8, 3],
                   mlp_ratio=[4, 4, 4, 4], kernel_size=7, f12_bn=True, g_bn=True, dwconv2_bn=True,
                   act=nn.ReLU6, downsampler_act=nn.Identity, block=g.Block)


def iformer_s(num_classes=3):
    f = _load_pkg("_r6b_iformer", "iFormer_ICLR2025/models", "iformer")
    return f.iFormer_s(num_classes=num_classes)


def lsnet_s(num_classes=3):
    _load_pkg("_r6b_lsnet", "LSNet_CVPR2025/model", "ska")                   # 相对导入依赖, 先显式加载
    l = _load_pkg("_r6b_lsnet", "LSNet_CVPR2025/model", "lsnet")
    _bmwc = l.build_model_with_cfg

    def _compat(model_cls, variant, pretrained=False, default_cfg=None, **kw):
        return _bmwc(model_cls, variant, pretrained=pretrained, **kw)

    l.build_model_with_cfg = _compat
    return l.lsnet_s(num_classes=num_classes)


REGISTRY = {n: ("adapters_r6b", n) for n in ("jumbo_small", "gmnet_s3", "iformer_s", "lsnet_s")}
