import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ultralytics import YOLO


CHECKPOINT = Path("/data/cle/sunfuxiang-1/ours/runs/detect/train48/weights/last.pt")


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {CHECKPOINT}")

    model = YOLO(str(CHECKPOINT))
    model.train(resume=True, batch=16)


if __name__ == "__main__":
    main()
