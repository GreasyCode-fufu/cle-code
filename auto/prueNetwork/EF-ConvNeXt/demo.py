"""EF-ConvNeXt 独立构建/推理演示(无其它项目依赖, 仅需 PyTorch/torchvision/PIL)。

用法:
    python demo.py                          # 构建模型 + 加载最优权重 + 示例前向
    python demo.py --image path/to/img.jpg  # 对单张 CLE 图像推理
    python demo.py --cpu                    # CPU 运行
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.optimize.convnext_ep import convnext_ef  # noqa: E402

CLASS_NAMES = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "weights", "ef_convnext_best.pth")
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="")
    ap.add_argument("--weights", type=str, default=WEIGHTS)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    model = convnext_ef(num_classes=3).to(device).eval()
    sd = torch.load(args.weights, map_location=device, weights_only=True)
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] EF-ConvNeXt  params={n:.2f}M (ConvNeXt-T: 28.06M, -34.3%)")

    if args.image:
        x = TF(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    else:
        x = torch.randn(1, 3, 224, 224, device=device)
        print("[demo] 未指定 --image, 使用随机张量演示前向")
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0]
    pred = int(prob.argmax())
    print(f"[output] logits shape=(1,3)")
    for c, p in zip(CLASS_NAMES, prob.tolist()):
        print(f"   {c:28s} {p:.4f}")
    print(f"[pred] {CLASS_NAMES[pred]} (conf={prob[pred]:.4f})")


if __name__ == "__main__":
    main()
