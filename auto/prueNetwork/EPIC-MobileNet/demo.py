#!/usr/bin/env python3
"""EPIC-MobileNet 演示: 构建 -> 前向(不依赖数据集); 可选加载训练好的权重对单张图分类。

  python demo.py                       # 随机权重: 结构/参数/FOV 掩码/前向自检
  python demo.py --cpu
  python demo.py --weights weights/mbv3_fz16_dce4m_seed42/mbv3_fz16_dce4m/model_best.pth --image some_cle.jpg
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from models.optimize.mbv3_cle_r6 import mbv3_dce4m, mbv3_r6_fz16, fov_weights  # noqa: E402

CLASSES = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def synthetic_cle(size=224):
    """方形画幅内嵌圆形视野的合成 CLE 样图: 视野外为 0(近黑背景), 视野内为随机纹理。"""
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    disk = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (0.48 * size) ** 2).float()
    img = (0.25 + 0.15 * torch.rand(3, size, size)) * disk
    return (img - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1), disk.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--weights", default="")
    ap.add_argument("--image", default="")
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")

    stage1 = mbv3_r6_fz16(num_classes=3)
    n_all = sum(p.numel() for p in stage1.parameters())
    n_tr = sum(p.numel() for p in stage1.parameters() if p.requires_grad)
    print(f"[stage 1] MobileNetV3-L + ① 证据保持冻结 /2–/16: 可训练 {n_tr/1e6:.3f}M / {n_all/1e6:.3f}M")

    net = mbv3_dce4m(num_classes=3).to(dev).eval()
    n = sum(p.numel() for p in net.parameters())
    print(f"[deploy ] EPIC-MobileNet(① 前段 + ② 凸证据头 + ③ FOV 感知四尺度证据): 参数 {n/1e6:.3f}M"
          f"(MobileNetV3-L baseline 4.206M), 证据维度 {net.classifier.fc.in_features}")

    x, fov_true = synthetic_cle()
    w = fov_weights(x.unsqueeze(0))
    print(f"[FOV    ] 合成样图圆形视野占比 {fov_true:.3f}, 掩码覆盖率 {w.mean().item():.3f}(1/4 分辨率 {tuple(w.shape[-2:])})")

    if a.weights:
        sd = torch.load(a.weights, map_location="cpu", weights_only=True)
        net.load_state_dict(sd)
        print(f"[weights] loaded {a.weights}")
    if a.image:
        from PIL import Image
        from torchvision import transforms
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                 transforms.Normalize(MEAN, STD)])
        x = tf(Image.open(a.image).convert("RGB"))
    with torch.no_grad():
        prob = torch.softmax(net(x.unsqueeze(0).to(dev)).float(), 1)[0].cpu()
    print("[forward] " + "  ".join(f"{c}: {p:.4f}" for c, p in zip(CLASSES, prob)))
    print("[demo OK]")


if __name__ == "__main__":
    main()
