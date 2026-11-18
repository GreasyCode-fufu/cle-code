#!/usr/bin/env python3
"""Clone-safe independent test-split evaluation for the frozen Gram RelKD best.pt."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

WEIGHTS = ROOT / "runs/detect/hongkong_SR1_P4Only_RelKD/weights/best.pt"
DATA = ROOT / "ultralytics/cfg/datasets/coco8.yaml"

if __name__ == "__main__":
    YOLO(str(WEIGHTS)).val(
        data=str(DATA),
        cfg=str(ROOT / "ultralytics/cfg/default.yaml"),
        split="test",
        batch=32,
        imgsz=640,
        workers=8,
        device=0,
        plots=False,
        project=str(ROOT / "runs/detect"),
        name="gram_relkd_clone_test",
    )

