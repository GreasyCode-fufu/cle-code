"""Register recent backbones, then delegate to the unchanged fair trainer."""
import sys
import runpy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latest_top_networks.adapters import REGISTRY
import cle_fair_common

cle_fair_common.MODEL_REGISTRY.update(REGISTRY)

if __name__ == "__main__":
    runpy.run_module("train_cle_fair", run_name="__main__")
