#!/usr/bin/env python3
"""GEM-ConvNeXt (convnext_v6f) 独立演示: 构建 -> (可选加载权重) -> 前向。

仅依赖 torch 与本包 models/; 无权重时用随机初始化(仅验证结构)。
用法: python demo.py [--cpu] [--weights path/to/model_best.pth]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.optimize.convnext_cle_v5 import convnext_v6f  # noqa: E402

CLASSES = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--weights", default="./weights/gem_convnext_best.pth")
    args = ap.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model = convnext_v6f(num_classes=len(CLASSES)).to(device).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[GEM-ConvNeXt] params={n_par / 1e6:.2f}M  device={device}")

    if os.path.exists(args.weights):
        sd = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(sd)
        print(f"[weights] loaded {args.weights}")
    else:
        print(f"[weights] {args.weights} 不存在, 使用随机初始化(仅结构演示)")

    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        y = model(x)
    probs = y.softmax(-1)[0]
    print(f"[forward] input={tuple(x.shape)} -> logits={tuple(y.shape)}")
    for c, p in zip(CLASSES, probs.tolist()):
        print(f"  {c:26s} p={p:.4f}")


if __name__ == "__main__":
    main()
