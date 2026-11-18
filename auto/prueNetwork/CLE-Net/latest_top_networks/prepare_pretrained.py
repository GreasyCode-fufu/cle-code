"""Materialize official Hub weights into auditable PyTorch state-dict files."""
import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path("/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks")
OUT = ROOT / "pretrained"


def save_model(model, path: Path, source: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "source": source}, path)
    print(f"saved {path} ({path.stat().st_size} bytes)")


def thinkingvit():
    repo = ROOT / "ThinkingViT_CVPR2026"
    sys.path.insert(0, str(repo))
    from timm.models import create_model

    source = "hf-hub:NCPS/thinkingvit_deit-3h-6h-imagenet1k"
    model = create_model(source, pretrained=True)
    save_model(model, OUT / "thinkingvit_deit_3h_6h_imagenet1k.pth", source)


def cpubone():
    import timm

    source = "cpubone_b2_bfrobust.r224_in1k"
    model = timm.create_model(source, pretrained=True)
    save_model(model, OUT / "cpubone_b2_bfrobust_r224_in1k.pth", f"timm:{source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("thinkingvit", "cpubone"))
    args = parser.parse_args()
    globals()[args.model]()
