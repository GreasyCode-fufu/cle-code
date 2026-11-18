#!/usr/bin/env python3
"""Evaluate one sweep checkpoint on an explicitly selected split."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/data/cle/sunfuxiang-2/ultralytics-main")
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from ultralytics import YOLO, __version__
from ultralytics.utils.torch_utils import get_flops


def scalar(x): return float(x.item() if hasattr(x, "item") else x)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--weights", required=True); p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--output", required=True); a = p.parse_args(); weight = Path(a.weights)
    extension_before = any(x == "ultralytics_ext" or x.startswith("ultralytics_ext.") for x in sys.modules)
    checkpoint = torch.load(weight, map_location="cpu", weights_only=False); checkpoint_model = checkpoint.get("model") or checkpoint.get("ema")
    if checkpoint_model is None: raise RuntimeError("checkpoint has no model")
    keys = list(checkpoint_model.state_dict()); forbidden = [x for x in keys if any(t in x.lower() for t in ("teacher", "discriminator", "composite", "attention_kd", "response_kd"))]
    model = YOLO(str(weight)); metrics = model.val(cfg="ultralytics/cfg/default.yaml", split=a.split, batch=32, imgsz=640, workers=8, device=0, plots=False)
    box, all_ap = metrics.box, metrics.box.all_ap; classes = {}
    for row, index in enumerate(box.ap_class_index.tolist()):
        classes[metrics.names[int(index)]] = {"precision": scalar(box.p[row]), "recall": scalar(box.r[row]), "ap50": scalar(box.ap50[row]), "ap75": scalar(all_ap[row, 5]), "ap50_95": scalar(box.ap[row])}
    fused = copy.deepcopy(checkpoint_model).float().fuse(verbose=False)
    result = {"status": "SUCCESS", "ultralytics_version": __version__, "weight_path": str(weight), "weight_size_bytes": weight.stat().st_size,
              "split": a.split, "overall": {"precision": scalar(box.mp), "recall": scalar(box.mr), "map50": scalar(box.map50), "map75": scalar(all_ap[:, 5].mean()), "map50_95": scalar(box.map)},
              "classes": classes, "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()},
              "unfused_params": sum(p.numel() for p in checkpoint_model.parameters()), "fused_params": sum(p.numel() for p in fused.parameters()),
              "gflops": float(get_flops(fused, imgsz=640)), "forbidden_checkpoint_keys": forbidden, "checkpoint_clean": not forbidden,
              "ultralytics_ext_loaded_before": extension_before,
              "inference_independent_of_ultralytics_ext": not extension_before and not any(x == "ultralytics_ext" or x.startswith("ultralytics_ext.") for x in sys.modules)}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
