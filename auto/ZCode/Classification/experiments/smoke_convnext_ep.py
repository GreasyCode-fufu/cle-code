import sys, time, torch
sys.path.insert(0, "/home/login/data/cle/auto/ZCode/Classification")
from cle_fair_common import build_model
from train_cle_fair import load_pretrained_weights_strict

PTH = "/home/login/data/cle/sunfuxiang/classifyML/pth/convnext_tiny_1k_224_ema.pth"
dev = torch.device("cuda:0")
torch.manual_seed(0)
x = torch.randn(4, 3, 224, 224, device=dev)
NAMES = ["convnext_dph", "convnext_fesr", "convnext_lsdg",
         "convnext_dph_fesr", "convnext_dph_lsdg", "convnext_fesr_lsdg",
         "convnext_dph_fesr_lsdg"]

for name in NAMES:
    m = build_model(name, num_classes=3).to(dev)
    load_pretrained_weights_strict(m, PTH, name)   # 严格审计: 仅白名单段缺失
    m.train()
    with torch.amp.autocast("cuda"):
        out = m(x); loss = out.sum()
    loss.backward()
    # 新增段梯度检查(不可旁路的必要条件: 梯度非零)
    newsegs = [n_ for n_, p in m.named_parameters()
               if any(s in n_ for s in ("dph", "fesr", "lsdg"))]
    gnorm = sum(p.grad.abs().sum().item() for n_, p in m.named_parameters()
                if p.grad is not None and any(s in n_ for s in ("dph", "fesr", "lsdg")))
    n_par = sum(p.numel() for p in m.parameters()) / 1e6
    n_new = sum(p.numel() for n_, p in m.named_parameters()
                if any(s in n_ for s in ("dph", "fesr", "lsdg"))) / 1e3
    m.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        t0 = time.time()
        for _ in range(10):
            m(x)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / 10 * 1000
    print(f"{name:24s} params={n_par:6.2f}M (+{n_new:6.1f}K) out={tuple(out.shape)} "
          f"new_grad={gnorm:.4f} fwd={ms:.1f}ms")
print("[EP SMOKE DONE]")
