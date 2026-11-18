"""R6 对比: 最新轻量网络(2024–2025 顶会, 官方代码与官方 ImageNet-1k 非蒸馏权重)—— 在本仓库统一协议下
与 EPIC-MobileNet 对比(训练/评测/种子/增广完全一致)。

代码库: latest_lightweight_nets/(git 浅克隆, 见其 MANIFEST.tsv; 只读, 不在其中执行)。
权重: 官方发布文件下载后以受限反序列化转存为纯张量 state_dict(pth/latest_r6/*.pth; lsnet_t/shvit_s1 取 EMA 副本——
与官方 val 评测口径一致)。

适配方式: 模块级加载官方文件(为相对导入建立独立合成包名, 不污染全局; MobileMamba 例外, 其代码用绝对包名
"model", 需把官方仓库根加入 sys.path —— 名字与本地 models/ 不冲突)。均为**官方实现零改动**, 仅包装工厂函数。
"""
import importlib.util
import sys
import types

REPO = "latest_lightweight_nets"
import os  # noqa: E402

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# MobileMamba 依赖 PyWavelets(官方 requirements 之外隐式依赖), 隔离安装在本仓库 comparison 目录内, 不污染环境
_PYDEPS = os.path.join(_ROOT, REPO, "_pydeps")
if os.path.isdir(_PYDEPS) and _PYDEPS not in sys.path:
    sys.path.insert(0, _PYDEPS)


def _load_pkg(pkg_name, rel_dir, filename):
    """把 <repo>/<rel_dir>/<filename> 加载为独立合成包 <pkg_name> 下的模块(支持其相对导入)。"""
    import os
    d = os.path.join(_ROOT, REPO, rel_dir)
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [d]
        sys.modules[pkg_name] = pkg
    if f"{pkg_name}.{filename}" in sys.modules:
        return sys.modules[f"{pkg_name}.{filename}"]
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.{filename}", os.path.join(d, filename + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# LSNet-T (CVPR 2025): 官方 lsnet_t 工厂。timm>=0.9 把 default_cfg 的 'hf_hub' 字段改名为 'hf_hub_id', 官方代码按
# timm 0.5 写的 @register_model 装饰器会在**导入期**构造 PretrainedCfg(hf_hub=...) 而报错 —— 在加载官方模块前给
# timm 的 PretrainedCfg 打兼容补丁(丢弃未知字段; 该字段仅用于 pretrained 下载, 我们 pretrained=False)。
import dataclasses  # noqa: E402
import timm.models._registry as _timm_reg  # noqa: E402

_PC_FIELDS = {f.name for f in dataclasses.fields(_timm_reg.PretrainedCfg)}


class _PretrainedCfgCompat(_timm_reg.PretrainedCfg):
    def __init__(self, **kw):
        super().__init__(**{k: v for k, v in kw.items() if k in _PC_FIELDS and k != "hf_hub"})


_timm_reg.PretrainedCfg = _PretrainedCfgCompat
import timm.models.registry as _timm_reg2  # noqa: E402
_timm_reg2.PretrainedCfg = _PretrainedCfgCompat

_lsnet = _load_pkg("_r6_lsnet", "LSNet_CVPR2025/model", "lsnet")
# timm>=0.9 的 build_model_with_cfg 也不再接受 default_cfg 入参(pretrained=False 时无需配置) —— 包装丢弃
_bmwc = _lsnet.build_model_with_cfg


def _bmwc_compat(model_cls, variant, pretrained=False, default_cfg=None, **kw):
    return _bmwc(model_cls, variant, pretrained=pretrained, **kw)


_lsnet.build_model_with_cfg = _bmwc_compat


def lsnet_t(num_classes=3):
    return _lsnet.lsnet_t(num_classes=num_classes)


# SHViT-S1 (CVPR 2024): 官方 build.py 工厂(shvit_s1)
_shvit = _load_pkg("_r6_shvit", "SHViT_CVPR2024/model", "shvit")
_shvit_build = _load_pkg("_r6_shvit", "SHViT_CVPR2024/model", "build")


def shvit_s1(num_classes=3):
    return _shvit_build.shvit_s1(num_classes=num_classes)


# iFormer-T (ICLR 2025): 官方 iFormer_t 工厂
_iformer = _load_pkg("_r6_iformer", "iFormer_ICLR2025/models", "iformer")


def iformer_t(num_classes=3):
    return _iformer.iFormer_t(num_classes=num_classes)


# EfficientViM-M1 (CVPR 2025): 官方 EfficientViM_M1 工厂(四尺度深监督头, 推理时按可学习权重融合)
_evim = _load_pkg("_r6_evim", "EfficientViM_CVPR2025/classification/models", "EfficientViM")


def efficientvim_m1(num_classes=3):
    return _evim.EfficientViM_M1(num_classes=num_classes)


# MobileMamba-T2 (CVPR 2025): 官方 MobileMamba_T2 工厂。**已放弃对比**(2026-11-09): 官方代码依赖需编译的
# selective_scan CUDA kernel; 未编译时纯 PyTorch 回退实测 >30 min/epoch(224², 4090), 训练不可行。
# 代码与权重保留作记录; 注册表已移除。
_MM_ROOT = os.path.join(_ROOT, REPO, "MobileMamba_CVPR2025")
if _MM_ROOT not in sys.path:
    sys.path.insert(0, _MM_ROOT)
_mm = importlib.import_module("model.mobilemamba.mobilemamba")


def mobilemamba_t2(num_classes=3):
    return _mm.MobileMamba_T2(num_classes=num_classes)
