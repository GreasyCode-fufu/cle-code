"""Register recent backbones, then delegate to the unchanged fair evaluator."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latest_top_networks.adapters import REGISTRY
import cle_fair_common

cle_fair_common.MODEL_REGISTRY.update(REGISTRY)

import evaluate

evaluate.MODEL_REGISTRY.update(REGISTRY)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CLE recent-network evaluation")
    p.add_argument("--model", required=True, choices=sorted(evaluate.MODEL_REGISTRY))
    p.add_argument("--weights", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--class-names", nargs="+", default=[
        "Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--norm-mean", type=float, nargs=3, default=None)
    p.add_argument("--norm-std", type=float, nargs=3, default=None)
    p.add_argument("--run-meta", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    evaluate.main(p.parse_args())
