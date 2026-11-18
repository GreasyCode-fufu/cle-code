"""R6 冒烟(MC 修订候选): star2x(扩展空间 /16 证据) 与 冻结变体。
检查: 严格加载; 主干逐位等价(含 block12 逐步展开实现 == 原 forward); 冻结段 requires_grad=False 且训练态下 BN 为推理态;
冻结变体初始(推理态)与 star2 逐位一致; 梯度只到可训练部分; 可训练参数量。"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/login/data/cle/auto/ZCode/Classification")
from cle_fair_common import build_model  # noqa: E402
from train_cle_fair import load_pretrained_weights_strict  # noqa: E402

PTH = "/home/login/data/cle/sunfuxiang/classifyML/pth/mobilenet_v3_large-8738ca79.pth"
dev = torch.device("cuda:0")


def build(name, seed=0):
    torch.manual_seed(seed)
    m = build_model(name, num_classes=3)
    load_pretrained_weights_strict(m, PTH, name)
    return m.to(dev)


torch.manual_seed(1)
x = torch.randn(8, 3, 224, 224, device=dev)
y = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0], device=dev)
base = build("mobilenet_v3_large").eval()
ref = build("mbv3_r6_star2").eval()
with torch.no_grad():
    f0 = base.features(x).mean(dim=(2, 3))
    out_ref = ref(x)

for name in ["mbv3_r6_star2x", "mbv3_r6_s2_fz8", "mbv3_r6_s2_fz16", "mbv3_r6_s2x_fz16"]:
    m = build(name).eval()
    with torch.no_grad():
        dfeat = (m.forward_pooled(x)["f16"] - f0).abs().max().item()     # 含 block12 展开实现
        dref = (m(x) - out_ref).abs().max().item() if m.v_src == "out" else float("nan")
    m.train()
    frozen_bn_eval = all(not mod.training for i in range(m.freeze_upto + 1)
                         for mod in m.features[i].modules() if isinstance(mod, nn.BatchNorm2d))
    with torch.amp.autocast("cuda"):
        loss = F.cross_entropy(m(x), y)
    loss.backward()
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    g_frozen = sum(p.grad.abs().sum().item() for i in range(m.freeze_upto + 1)
                   for p in m.features[i].parameters() if p.grad is not None)
    g_head = sum(p.grad.abs().sum().item() for p in m.classifier.parameters() if p.grad is not None)
    g_top = sum(p.grad.abs().sum().item() for p in m.features[15].parameters() if p.grad is not None)
    print(f"{name:18s} params={sum(p.numel() for p in m.parameters())/1e6:.3f}M trainable={n_tr/1e6:.3f}M "
          f"max|Δf16|={dfeat:.1e} max|Δlogits vs star2|={dref:.1e} frozenBN_eval={frozen_bn_eval} "
          f"g_frozen={g_frozen:.1e} g_head={g_head:.1e} g_top={g_top:.1e}")
    assert dfeat == 0.0 and g_head > 0 and g_top > 0 and g_frozen == 0.0 and frozen_bn_eval
    if m.v_src == "out":
        assert dref == 0.0
print("[R6 MC SMOKE PASSED]")
