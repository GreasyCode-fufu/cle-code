"""创新点模型冒烟测试:
 1. 全部工厂可构建, AMP 前向反向正常
 2. 预训练严格加载(白名单段) 通过
 3. 功能等价性: 零门控 => convnext_wds_wtb_msf 加载预训练后初始输出
    应与原版 convnext_tiny 完全一致(逐位/近逐位) —— 保证创新点是"无害增强"
 4. 参数量增量报告
"""
import sys, torch
sys.path.insert(0, "/home/login/data/cle/auto/ZCode/Classification")

from cle_fair_common import build_model
from train_cle_fair import load_pretrained_weights_strict

PTH = "/home/login/data/cle/sunfuxiang/classifyML/pth/convnext_tiny_1k_224_ema.pth"
dev = torch.device("cuda:0")
NAMES = ["convnext_wds", "convnext_wtb", "convnext_msf",
         "convnext_wds_wtb", "convnext_wds_wtb_msf",
         "convnext_wds_wtb_l1", "convnext_wds_wtb_s234", "convnext_wds_wtb_k3", "convnext_wtb_k3", "convnext_wds_wtb_k3_msf", "convnext_wdd", "convnext_wds_msf", "convnext_wtb_msf", "convnext_wds_wdd_msf"]

torch.manual_seed(0)
x = torch.randn(4, 3, 224, 224, device=dev)

# 原版参考输出(主干特征; 分类头被加载器剔除保持随机, 比较logits无意义)
ref = build_model("convnext_tiny", num_classes=3).to(dev)
load_pretrained_weights_strict(ref, PTH, "convnext_tiny")
ref.eval()
with torch.no_grad():
    y0 = ref.forward_features(x)
n_ref = sum(p.numel() for p in ref.parameters())

for name in NAMES:
    m = build_model(name, num_classes=3).to(dev)
    load_pretrained_weights_strict(m, PTH, name)
    m.eval()
    with torch.no_grad():
        y1 = m.forward_features(x)
    eq = torch.allclose(y0, y1, atol=1e-5)
    md = (y0 - y1).abs().max().item()
    n = sum(p.numel() for p in m.parameters())
    # 训练一步(AMP) 确保新参数有梯度
    m.train()
    with torch.amp.autocast("cuda"):
        loss = m(x).sum()
    loss.backward()
    gok = all(p.grad is not None for n_, p in m.named_parameters()
              if p.requires_grad and ("wtg" in n_ or "wds_gate" in n_ or "msf_g" in n_))
    print(f"{name:24s} params={n/1e6:7.2f}M (+{(n-n_ref)/1e6:.2f}M)  "
          f"init_equal={eq}(maxdiff={md:.2e})  gates_grad={gok}")
    assert eq and gok, f"{name} FAILED"

print("\n[ALL INNOVATION SMOKE TESTS PASSED]")
