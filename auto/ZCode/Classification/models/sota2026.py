"""R6-B 对比任务: CLE-Net(convnext_egb_scg) vs 最新顶会网络(Jumbo/GmNet ICLR 2026, iFormer-S ICLR 2025,
LSNet-S CVPR 2025)—— 官方实现零改动接入本仓库统一协议。

代码库: sota_nets_2026/(各网络独立子文件夹, 见 MANIFEST.tsv; 与 Codex 的 CLE-Net/latest_top_networks 任务相互独立)。
权重: 用户手动下载上传(sota2026_upload/, HF/GitHub/GDrive 官方发布), 经受限反序列化转存为纯张量(pth/sota2026/*.pth;
GmNet 官方以 np.save(state_dict) 发布 → 从 npy 重建张量)。

适配: 合成包加载(相对导入); timm≥0.9 兼容补丁(LSNet 的 hf_hub/default_cfg 用法)。四个网络的分类头根名均在
CLASSIFIER_ROOTS 内(jumbo/gmnet/lsnet: head; iformer: classifier), 严格加载审计无需改动。
"""
import dataclasses
import importlib.util
import os
import sys
import types

import torch

REPO = "sota_nets_2026"
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _load_pkg(pkg_name, rel_dir, filename):
    """把 <repo>/<rel_dir>/<filename> 加载为独立合成包(支持其相对导入), 不污染全局命名。"""
    d = os.path.join(_ROOT, REPO, rel_dir)
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


# ── timm≥0.9 兼容补丁(LSNet 按 timm 0.5 写法: @register_model 在导入期构造 PretrainedCfg(hf_hub=...),
#    build_model_with_cfg 仍传 default_cfg) —— 在加载 LSNet 前打补丁, 不改动官方文件 ──
import timm.models._registry as _timm_reg  # noqa: E402

if getattr(_timm_reg.PretrainedCfg, "__module__", "") != __name__:
    _PC_FIELDS = {f.name for f in dataclasses.fields(_timm_reg.PretrainedCfg)}

    class _PretrainedCfgCompat(_timm_reg.PretrainedCfg):
        def __init__(self, **kw):
            super().__init__(**{k: v for k, v in kw.items() if k in _PC_FIELDS and k != "hf_hub"})

    _timm_reg.PretrainedCfg = _PretrainedCfgCompat
    import timm.models.registry as _timm_reg2
    _timm_reg2.PretrainedCfg = _PretrainedCfgCompat


# ── Jumbo-small (ICLR 2026, "Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers") ──
# plain ViT + 共享 jumbo-CLS MLP; 官方 eval_i1k.py 的构建方式: get_config(size) → Jumbo(gpu_id, J, **kwargs)
_jumbo = _load_pkg("_r6b_jumbo", "Jumbo_ICLR2026/models", "jumbo")
_jumbo_cfg = _load_pkg("_r6b_jumbo", "Jumbo_ICLR2026", "configs")


def jumbo_small(num_classes=3):
    kwargs = _jumbo_cfg.get_config("small")
    J = kwargs.pop("J", None)
    kwargs["num_classes"] = num_classes
    return _jumbo.Jumbo(torch.device("cpu"), J, **kwargs)


# ── GmNet-s3 (ICLR 2026, "GmNet: Revisiting Gating Mechanisms From A Frequency Perspective") ──
# 官方发布的 gmnet_s3 ImageNet checkpoint 含 f1.bn / dwconv2.bn(各 17 块), 而仓库工厂默认 f12_bn=False,
# dwconv2_bn=False —— 按 checkpoint 的实际结构构建(其余超参与官方 gmnet_s3 工厂逐项一致), 保证 100% 加载。
_gmnet = _load_pkg("_r6b_gmnet", "GmNet_ICLR2026", "gment")


def gmnet_s3(num_classes=3):
    import torch.nn as nn
    return _gmnet.Model(num_classes=num_classes, embed_dim=[48, 96, 192, 384], depths=[3, 3, 8, 3],
                        mlp_ratio=[4, 4, 4, 4], kernel_size=7, f12_bn=True, g_bn=True, dwconv2_bn=True,
                        act=nn.ReLU6, downsampler_act=nn.Identity, block=_gmnet.Block)


# ── iFormer-S (ICLR 2025) ──
_iformer = _load_pkg("_r6b_iformer", "iFormer_ICLR2025/models", "iformer")


def iformer_s(num_classes=3):
    return _iformer.iFormer_s(num_classes=num_classes)


# ── LSNet-S (CVPR 2025) ──
_lsnet = _load_pkg("_r6b_lsnet", "LSNet_CVPR2025/model", "lsnet")
_bmwc = _lsnet.build_model_with_cfg


def _bmwc_compat(model_cls, variant, pretrained=False, default_cfg=None, **kw):
    return _bmwc(model_cls, variant, pretrained=pretrained, **kw)


_lsnet.build_model_with_cfg = _bmwc_compat


def lsnet_s(num_classes=3):
    return _lsnet.lsnet_s(num_classes=num_classes)
