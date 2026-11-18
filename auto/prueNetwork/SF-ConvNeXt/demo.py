"""SF-ConvNeXt 独立构建/推理演示(无其它项目依赖, 仅需 PyTorch)。

用法:
    python demo.py                # 构建 + 加载最优权重 + 前向演示
    python demo.py --cpu          # CPU 上运行
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.optimize.convnext_cle_v2 import convnext_v2bc  # noqa: E402

# 类别顺序与训练时一致(sorted 目录序)
CLASS_NAMES = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "weights", "sf_convnext_best.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")

    # 1. 构建 SF-ConvNeXt (= ConvNeXt-T + WR + CH)
    model = convnext_v2bc(num_classes=3).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[build] SF-ConvNeXt OK  params={n_par:.2f}M")

    # 2. 加载最优 checkpoint(seed42, test Macro-F1 0.9672)
    if os.path.exists(WEIGHTS):
        sd = torch.load(WEIGHTS, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not [k for k in missing if k != ""], f"missing: {missing[:5]}"
        assert not unexpected, f"unexpected: {unexpected[:5]}"
        print(f"[load ] checkpoint OK -> {WEIGHTS}")
    else:
        print(f"[load ] 未找到权重(随机初始化): {WEIGHTS}")

    # 3. 推理演示
    model.eval()
    x = torch.randn(2, 3, 224, 224, device=device)  # 实际使用时替换为
    # 归一化后的 CLE 图像张量: Resize(256)->CenterCrop(224)->ToTensor->Normalize(
    #     mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])   # ImageNet 统计
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    for i, p in enumerate(probs):
        pred = CLASS_NAMES[int(p.argmax())]
        print(f"[infer] 样本{i}: 预测={pred}  概率={p.tolist()}")


if __name__ == "__main__":
    main()
