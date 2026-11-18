import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latest_top_networks.adapters import REGISTRY
import cle_fair_common
from train_cle_fair import load_pretrained_weights_strict

cle_fair_common.MODEL_REGISTRY.update(REGISTRY)

weights = {
    "thinkingvit_deit": "thinkingvit_deit_3h_6h_imagenet1k.pth",
    "cpubone_b2": "cpubone_b2_bfrobust_r224_in1k.normalized.pth",
    "mambavision_t": "mambavision_tiny_1k.normalized.pth",
    "mambaout_tiny": "mambaout_tiny.normalized.pth",
    "transnext_tiny": "transnext_tiny_224_1k.pth",
}
name = sys.argv[1]
model = cle_fair_common.build_model(name, 3)
path = ROOT / "latest_top_networks" / "pretrained" / weights[name]
info = load_pretrained_weights_strict(model, path, name)
print(name, info)
