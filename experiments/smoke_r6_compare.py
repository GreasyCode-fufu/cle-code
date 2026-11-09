"""R6 冒烟(实验 1/2): 严格加载审计 + 冻结语义 + 初始化等价 + 前向/反向。
用法: python experiments/smoke_r6_compare.py"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/login/data/cle/auto/ZCode/Classification")
from cle_fair_common import build_model  # noqa: E402
from train_cle_fair import load_pretrained_weights_strict  # noqa: E402

P = "/home/login/data/cle/sunfuxiang/classifyML/pth/"
PTH = {"mbv3": P + "mobilenet_v3_large-8738ca79.pth", "convnext_tiny": P + "convnext_tiny_1k_224_ema.pth",
       "starnet_s2": P + "timm_r6/starnet_s2.pth", "fast_t0": P + "fasternet_t0-epoch.281-val_acc1.71.9180.pth"}
dev = torch.device("cuda:0")
torch.manual_seed(1)
x = torch.randn(4, 3, 224, 224, device=dev)
y = torch.tensor([0, 1, 2, 1], device=dev)


def fwd_bwd(m):
    m.train()
    with torch.amp.autocast("cuda"):
        loss = F.cross_entropy(m(x), y)
        aux = getattr(m, "mfm", None)
        if aux is not None and aux.loss is not None:
            loss = loss + m.mfm_lambda * aux.loss
    loss.backward()
    return loss.item()


def bn_modes(mods):
    return {b.training for mod in mods for b in mod.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)}


print("== 实验 1 ==")
base = build_model("mobilenet_v3_large", 3)
load_pretrained_weights_strict(base, PTH["mbv3"], "mobilenet_v3_large")
base = base.to(dev).eval()
with torch.no_grad():
    f0 = base.features(x).mean((2, 3))
# LP-FT: 注入的头 = 线性探针(在原始特征空间)
m = build_model("mbv3_lpft", 3)
a = load_pretrained_weights_strict(m, PTH["mbv3"], "mbv3_lpft")
assert a["missing_keys"] == [] and not a["unexpected_keys"]
head = torch.load("/home/login/data/cle/auto/ZCode/Classification/experiments/r6_lpft_head_cle.pt")
m = m.to(dev).eval()
with torch.no_grad():
    d = (m(x) - (f0 @ head["weight"].to(dev).T + head["bias"].to(dev))).abs().max().item()
print(f"mbv3_lpft: head injected (missing=0), logits == LP(f16) max|Δ|={d:.1e}, loss={fwd_bwd(m):.3f}")
assert d < 1e-3
# L2-SP: 起点 = 预训练; 初始 SP 项为 0
for n in ("mbv3_l2sp_a01", "mbv3_l2sp_a001"):
    m = build_model(n, 3)
    a = load_pretrained_weights_strict(m, PTH["mbv3"], n)
    m = m.to(dev)
    same = all(torch.equal(p, r) for p, r in zip(m.features.parameters(), m.sp_ref.parameters()))
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    loss = fwd_bwd(m)
    l_sp = sum(((p - r) ** 2).sum() for p, r in zip(m.features.parameters(), m.sp_ref.parameters()) if p.dim() > 1).item()
    print(f"{n}: sp_ref==pretrained {same}, SP term@init {l_sp:.1e}, aux(β·new)={m._aux.item():.3f}, trainable {n_tr/1e6:.3f}M, loss {loss:.3f}")
    assert same and l_sp == 0.0 and abs(n_tr - 4205875) < 10
# Surgical: 只有所选块 + 头可训; BN 训练态(官方)
for n, rng in (("mbv3_sft_b1", range(0, 4)), ("mbv3_sft_b2", range(4, 7)), ("mbv3_sft_b3", range(7, 13)),
               ("mbv3_sft_b4", range(13, 17)), ("mbv3_sft_head", range(0))):
    m = build_model(n, 3)
    load_pretrained_weights_strict(m, PTH["mbv3"], n)
    m = m.to(dev)
    fwd_bwd(m)
    tuned = [i for i, blk in enumerate(m.features) if any(p.requires_grad for p in blk.parameters())]
    graded = [i for i, blk in enumerate(m.features) if any(p.grad is not None and p.grad.abs().sum() > 0 for p in blk.parameters())]
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{n}: tuned blocks {tuned} graded {graded}, BN train-mode {bn_modes(m.features)}, trainable {n_tr/1e6:.3f}M")
    assert tuned == list(rng) == graded and bn_modes(m.features) == {True}

print("== 实验 2 ==")
for arch in ("convnext_tiny", "starnet_s2", "fast_t0"):
    torch.manual_seed(42)
    v = build_model(arch, 3)
    torch.manual_seed(42)
    f = build_model(f"{arch}_fz3", 3)
    same_init = all(torch.equal(p, q) for p, q in zip(v.state_dict().values(), f.state_dict().values()))
    a = load_pretrained_weights_strict(f, PTH[arch], f"{arch}_fz3")
    from models.optimize.epic_generic import _front_modules
    front = _front_modules(arch, f)
    f = f.to(dev)
    fwd_bwd(f)
    n_tr = sum(p.numel() for p in f.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in f.parameters())
    g_front = sum(p.grad.abs().sum().item() for mod in front for p in mod.parameters() if p.grad is not None)
    print(f"{arch}_fz3: init==vanilla {same_init}, strict missing={a['missing_keys']}, trainable {n_tr/1e6:.3f}/{n_all/1e6:.3f}M, "
          f"front grad {g_front:.1e}, front BN modes {bn_modes(front) or 'n/a(LN)'}")
    assert same_init and g_front == 0.0 and bn_modes(front) in (set(), {False})
    # 阶段 2: load_stage1 + 前向 + 证据维度; 无暗背景时 mgap == gap
    sd = {k: t.detach().cpu() for k, t in f.state_dict().items()}
    for dst in (f"{arch}_fz3_dce4m", f"{arch}_lp_dce4s", f"{arch}_fz3_dce_last"):
        e = build_model(dst, 3)
        e.load_stage1(sd)
        e = e.to(dev).eval()
        with torch.no_grad():
            out = e(x)
        print(f"   {dst}: out {tuple(out.shape)} evidence dim {e.classifier.fc.in_features} deploy params "
              f"{sum(p.numel() for p in e.parameters())/1e6:.3f}M")
    bright = torch.full((2, 3, 224, 224), 1.0, device=dev)
    e1, e2 = build_model(f"{arch}_dce4m", 3), build_model(f"{arch}_lp_dce4s", 3)
    e1.load_stage1(sd); e2.load_stage1(sd)
    e2.classifier.load_state_dict(e1.classifier.state_dict())
    with torch.no_grad():
        dd = (e1.to(dev).eval()(bright) - e2.to(dev).eval()(bright)).abs().max().item()
    print(f"   no-dark-background: mgap == gap max|Δ|={dd:.1e}")
    assert dd < 1e-4
print("[R6 COMPARE SMOKE PASSED]")
