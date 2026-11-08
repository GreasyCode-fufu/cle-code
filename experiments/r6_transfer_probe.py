#!/usr/bin/env python3
"""R6 可迁移性探针: 冻结 ImageNet mbv3 四尺度 GAP 特征 + 强 L2 逻辑回归(C 仅在 val 上选)的 val/test Macro-F1。
用途: 检验"冻结预训练证据(①)在哪些域有益"能否由只用 train/val 的探针预判(LP-FT 论: 线性探针接近全量微调时,
微调主要带来特征失真)。与各域 baseline 全量微调的 val F1 对照。
用法: python experiments/r6_transfer_probe.py busi heel covid cub tcga cle
输出: experiments/r6_transfer_probe.json(按域合并)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import json
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torchvision import transforms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from models.MobileNetV3 import mobilenet_v3_large  # noqa: E402

PTH = "/home/login/data/cle/sunfuxiang/classifyML/pth/mobilenet_v3_large-8738ca79.pth"
DATA = {"cle": "datasets/cle_ac_v1", "busi": "datasets/busi_v1", "heel": "datasets/heelBone",
        "covid": "datasets/covid_v1", "cub": "datasets/cub_v1", "tcga": "datasets/TCGA"}
OUT = os.path.join(REPO, "experiments", "r6_transfer_probe.json")
TAPS = (3, 6, 12, 16)
CGRID = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class DS(torch.utils.data.Dataset):
    def __init__(self, root):
        cls = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.items = [(os.path.join(root, c, f), i) for i, c in enumerate(cls)
                      for f in sorted(os.listdir(os.path.join(root, c))) if f.lower().endswith(EXTS)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return TF(Image.open(p).convert("RGB")), y


@torch.no_grad()
def feats(m, root):
    acts = {}
    hs = [m.features[i].register_forward_hook(lambda mm, inp, o, k=i: acts.__setitem__(k, o)) for i in TAPS]
    X, Y = [], []
    for x, y in torch.utils.data.DataLoader(DS(root), batch_size=128, num_workers=12):
        m(x.cuda())
        X.append(torch.cat([acts[i].float().mean((2, 3)) for i in TAPS], 1).cpu())
        Y.append(y)
    for h in hs:
        h.remove()
    return torch.cat(X).numpy(), torch.cat(Y).numpy()


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    m = mobilenet_v3_large(num_classes=1000)
    m.load_state_dict(torch.load(PTH, map_location="cpu"))
    m.eval().cuda()
    for dom in sys.argv[1:]:
        d = os.path.join(REPO, DATA[dom])
        (Xtr, ytr), (Xva, yva), (Xte, yte) = (feats(m, os.path.join(d, sp)) for sp in ("train", "val", "test"))
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-8] = 1
        Ztr, Zva, Zte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
        best = None
        for C in CGRID:
            clf = LogisticRegression(C=C, max_iter=3000).fit(Ztr, ytr)
            f = f1_score(yva, clf.predict(Zva), average="macro")
            if best is None or f > best[0]:
                best = (f, C, clf)
        te = f1_score(yte, best[2].predict(Zte), average="macro")
        res[dom] = {"probe_val_f1": round(best[0] * 100, 2), "probe_test_f1": round(te * 100, 2), "C": best[1]}
        print(f"{dom:6s} frozen-ImageNet 4-scale probe: val F1 {best[0]*100:.2f}  test F1 {te*100:.2f}  (C={best[1]})", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
