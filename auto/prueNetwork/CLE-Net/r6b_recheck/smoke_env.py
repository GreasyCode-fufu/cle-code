"""cle_topnets 环境冒烟: 构建 4 网络 + 加载官方权重 + GPU 前向(严格加载审计逻辑同 train_cle_fair)。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adapters_r6b
import torch
P = "/home/login/data/cle/sunfuxiang/classifyML/pth/sota2026/"
dev = torch.device("cuda")
for name in ("jumbo_small", "gmnet_s3", "iformer_s", "lsnet_s"):
    m = getattr(adapters_r6b, name)(num_classes=3)
    sd = torch.load(P + name + ".pth", map_location="cpu")
    sd = {k: v for k, v in sd.items() if k.split(".")[0] not in ("head", "classifier")}  # 训练入口同款: 剥离分类头
    inc = m.load_state_dict(sd, strict=False)
    assert not inc.unexpected_keys and all(k.split(".")[0] in ("head", "classifier") for k in inc.missing_keys), (name, inc)
    m = m.to(dev).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224, device=dev))
    print(f"{name:12s} out {tuple(out.shape)} params {sum(p.numel() for p in m.parameters())/1e6:.3f}M missing={len(inc.missing_keys)} unexpected=0")
print("[R6-B ENV SMOKE PASSED]")
