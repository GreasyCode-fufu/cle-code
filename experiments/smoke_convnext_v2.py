import sys, torch
sys.path.insert(0, "/home/login/data/cle/auto/ZCode/Classification")
from cle_fair_common import build_model
from train_cle_fair import load_pretrained_weights_strict

PTH = "/home/login/data/cle/sunfuxiang/classifyML/pth/convnext_tiny_1k_224_ema.pth"
dev = torch.device("cuda:0")
torch.manual_seed(0)
x = torch.randn(4, 3, 224, 224, device=dev)
ref = build_model("convnext_tiny", num_classes=3).to(dev)
load_pretrained_weights_strict(ref, PTH, "convnext_tiny"); ref.eval()
with torch.no_grad():
    y0 = ref.forward_features(x)

for name in ["convnext_v2a", "convnext_v2b", "convnext_v2c", "convnext_v2ab", "convnext_v2abc"]:
    m = build_model(name, num_classes=3).to(dev)
    load_pretrained_weights_strict(m, PTH, name)
    m.train()
    with torch.amp.autocast("cuda"):
        out = m(x); loss = out.sum()
    loss.backward()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    f = m.eval().forward_features(x).flatten().float().cpu().detach()
    g = y0.flatten().cpu()
    k = min(len(f), len(g))
    corr = torch.corrcoef(torch.stack([f[:k], g[:k]]))[0, 1].item()
    print(f"{name:18s} params={n:6.2f}M out={tuple(out.shape)} feat_corr_with_vanilla={corr:.3f}")
print("[V2 SMOKE DONE]")
