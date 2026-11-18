import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latest_top_networks.adapters import REGISTRY
from cle_fair_common import MODEL_REGISTRY, build_model

MODEL_REGISTRY.update(REGISTRY)
name = sys.argv[1]
model = build_model(name, 3).cuda().eval()
with torch.inference_mode(), torch.amp.autocast("cuda"):
    y = model(torch.randn(1, 3, 224, 224, device="cuda"))
roots = sorted({key.split(".", 1)[0] for key in model.state_dict()})
print(name, tuple(y.shape), sum(p.numel() for p in model.parameters()), roots)
