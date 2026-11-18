import importlib.util
import os
import sys
import traceback
import types

import torch


ROOT = "/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks"


def probe(name, factory):
    try:
        model = factory().eval().cuda()
        params = sum(p.numel() for p in model.parameters())
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            output = model(torch.randn(1, 3, 224, 224, device="cuda"))
        if isinstance(output, dict):
            output = output.get("logits", next(iter(output.values())))
        print(f"OK\t{name}\tparams={params}\tshape={tuple(output.shape)}", flush=True)
    except Exception as exc:
        print(f"FAIL\t{name}\t{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()


def mambaout():
    repo = os.path.join(ROOT, "MambaOut_CVPR2025")
    sys.path.insert(0, repo)
    from models.mambaout import mambaout_tiny
    return mambaout_tiny(pretrained=False, num_classes=3)


def mambavision():
    repo = os.path.join(ROOT, "MambaVision_CVPR2025")
    sys.path.insert(0, repo)
    from mambavision.models.mamba_vision import mamba_vision_T
    return mamba_vision_T(pretrained=False, num_classes=3)


def transnext():
    repo = os.path.join(ROOT, "TransNeXt_CVPR2024", "classification")
    sys.path.insert(0, repo)
    from transnext import transnext_tiny
    return transnext_tiny(pretrained=False, num_classes=3)


def vmamba():
    repo = os.path.join(ROOT, "VMamba_NeurIPS2024", "classification")
    sys.path.insert(0, repo)
    from models.vmamba import VSSM
    return VSSM(
        patch_size=4,
        in_chans=3,
        num_classes=3,
        depths=[2, 2, 8, 2],
        dims=96,
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz",
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
    )


def unireplknet():
    repo = os.path.join(ROOT, "UniRepLKNet_CVPR2024")
    sys.path.insert(0, repo)
    from unireplknet import unireplknet_t
    return unireplknet_t(in_1k_pretrained=False, num_classes=3)


def cpubone():
    repo = os.path.join(ROOT, "CPUBone_CVPR2026")
    old_cwd = os.getcwd()
    sys.path.insert(0, repo)
    os.chdir(repo)
    try:
        from cpubone_model import get_cpubone_b2
        model = get_cpubone_b2(pretrained=False)
    finally:
        os.chdir(old_cwd)
    if (hasattr(model, "head") and hasattr(model.head, "op_list")
            and hasattr(model.head.op_list[3], "linear")):
        old = model.head.op_list[3].linear
        model.head.op_list[3].linear = torch.nn.Linear(old.in_features, 3)
    elif hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):
        model.head = torch.nn.Linear(model.head.in_features, 3)
    elif hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear):
        model.classifier = torch.nn.Linear(model.classifier.in_features, 3)
    else:
        raise RuntimeError(f"unknown CPUBone classifier: {model}")
    return model


class _ThinkingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x, threshold=0.0)
        if isinstance(output, tuple):
            # During training the official implementation returns stage logits;
            # during evaluation it returns (adaptive_logits, stage_mask).
            return output[-1] if self.model.training else output[0]
        return output


def thinkingvit():
    repo = os.path.join(ROOT, "ThinkingViT_CVPR2026")
    # The official repository ships a modified timm package required by ThinkingViT.
    sys.path.insert(0, repo)
    for key in [k for k in list(sys.modules) if k == "timm" or k.startswith("timm.")]:
        del sys.modules[key]
    from timm.models import create_model
    model = create_model("thinkingvit", pretrained=False, num_classes=3, thinking_stages=[3, 6])
    # Official entropy helper assumes ImageNet-style C>=10; make it valid for CLE's
    # three logits without changing the network or routing rule.
    def _entropy(this, logits):
        probs = torch.softmax(logits, dim=1)
        topk = torch.topk(probs, min(10, probs.shape[1]), dim=1)[0]
        return torch.distributions.Categorical(probs=topk).entropy().unsqueeze(1)
    model.entropy = types.MethodType(_entropy, model)
    return _ThinkingWrapper(model)


if __name__ == "__main__":
    factories = {
        "mambaout_tiny": mambaout,
        "mambavision_t": mambavision,
        "transnext_tiny": transnext,
        "vmamba_tiny": vmamba,
        "unireplknet_t": unireplknet,
        "cpubone_b2": cpubone,
        "thinkingvit": thinkingvit,
    }
    selected = sys.argv[1:] or list(factories)
    for model_name in selected:
        probe(model_name, factories[model_name])
