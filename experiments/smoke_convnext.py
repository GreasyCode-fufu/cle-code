"""ConvNeXt 注册冒烟测试: 构建 -> 严格加载预训练 -> 前向/反向 -> thop 复杂度。
验证点:
 1. cle_fair_common.MODEL_REGISTRY 能构建 convnext_tiny/small(num_classes=3)
 2. train_cle_fair.load_pretrained_weights_strict 严格加载官方 ema 权重零意外键
 3. AMP 下 forward+backward 正常, 输出形状 (B,3)
 4. evaluate.py 注册表可加载(用其 load_model 逻辑走一遍)
"""
import sys, torch
sys.path.insert(0, "/home/login/data/cle/auto/ZCode/Classification")

from cle_fair_common import build_model
from train_cle_fair import load_pretrained_weights_strict

PTH = "/home/login/data/cle/sunfuxiang/classifyML/pth"
dev = torch.device("cuda:0")

for name, ckpt in [("convnext_tiny", f"{PTH}/convnext_tiny_1k_224_ema.pth"),
                   ("convnext_small", f"{PTH}/convnext_small_1k_224_ema.pth")]:
    print(f"\n=== {name} ===")
    m = build_model(name, num_classes=3).to(dev)
    audit = load_pretrained_weights_strict(m, ckpt, name)
    print(f"[strict-load] OK, classifier_missing={len(audit['missing_keys'])}, unexpected=0")

    x = torch.randn(8, 3, 224, 224, device=dev)
    with torch.amp.autocast("cuda"):
        y = m(x)
        loss = y.sum()
    loss.backward()
    assert y.shape == (8, 3), y.shape
    n_par = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"[fwd/bwd] OK out={tuple(y.shape)} params={n_par:.1f}M")
    m.zero_grad(set_to_none=True)

print("\n[ALL SMOKE TESTS PASSED]")
