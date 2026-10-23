#!/usr/bin/env python3
"""eval_plus: 涨点手段评估器(用户已全部批准, 2026-10-24)

手段: TTA(8视图二面体平均) + 类别校准(val拟合逐类乘子) + 多模型概率集成。
校准只在 val 拟合, test 仅最终应用。

用法:
  python experiments/eval_plus.py --ckpt name=ckpt_path --ckpt name2=path2 ...
输出: 追加 experiments/results.csv (batch=evalplus) + 写 shared/leaderboard.md
"""
import argparse, csv, json, os, sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cle_fair_common import build_model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_split(root):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    paths, labels = [], []
    for i, c in enumerate(classes):
        cdir = os.path.join(root, c)
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(EXTS):
                paths.append(os.path.join(cdir, fn))
                labels.append(i)
    return paths, np.array(labels), classes


@torch.no_grad()
def infer(model, paths, img_size, mean, std, tta=True, bs=64):
    """8 视图二面体 TTA(4旋转x2翻转); tta=False 为单视图。返回平均 softmax。"""
    views = [(k, fl) for k in range(4) for fl in (False, True)] if tta else [(0, False)]
    probs_sum = None
    for k, fl in views:
        def tf(img, k=k, fl=fl):
            if k:
                img = img.rotate(90 * k, Image.BILINEAR)
            if fl:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return img
        t = transforms.Compose([
            transforms.Lambda(tf),
            transforms.Resize((int(img_size * 1.143),) * 2),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(), transforms.Normalize(mean, std)])
        out, batch = [], []
        for p in paths:
            batch.append(t(Image.open(p).convert("RGB")))
            if len(batch) == bs:
                x = torch.stack(batch).to(DEVICE)
                out.append(F.softmax(model(x), dim=1).cpu().numpy())
                batch = []
        if batch:
            x = torch.stack(batch).to(DEVICE)
            out.append(F.softmax(model(x), dim=1).cpu().numpy())
        p_one = np.concatenate(out)
        probs_sum = p_one if probs_sum is None else probs_sum + p_one
    return probs_sum / len(views)


def calibrate(val_probs, val_y, n_classes, passes=2, grid=np.arange(0.75, 1.31, 0.025)):
    """坐标上升拟合逐类乘子, 最大化 val macro-F1。"""
    w = np.ones(n_classes)

    def f1_of(w):
        return f1_score(val_y, (val_probs * w).argmax(1), average="macro", zero_division=0)

    for _ in range(passes):
        improved = False
        for c in range(n_classes):
            best, cur = w[c], f1_of(w)
            for g in grid:
                w2 = w.copy(); w2[c] = g
                s = f1_of(w2)
                if s > cur + 1e-6:
                    best, cur = g, s
            if best != w[c]:
                w[c] = best
                improved = True
        if not improved:
            break
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    args = ap.parse_args()
    entries = [tuple(s.split("=", 1)) for s in args.ckpt]

    rows, store = [], {}
    for name, ckpt in entries:
        model = build_model(name, num_classes=3)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd, strict=True)
        model.to(DEVICE).eval()
        meta_p = os.path.join(os.path.dirname(ckpt), "run_meta.json")
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        img_size = meta.get("img_size", 224)
        mean = meta.get("norm_mean", [0.485, 0.456, 0.406])
        std = meta.get("norm_std", [0.229, 0.224, 0.225])
        vp, vy, _ = load_split(os.path.join(REPO, "datasets/cle/val"))
        tp, ty, _ = load_split(os.path.join(REPO, "datasets/cle/test"))
        v_p = infer(model, vp, img_size, mean, std)
        t_p = infer(model, tp, img_size, mean, std)
        store[name] = dict(v=v_p, vy=vy, t=t_p, ty=ty)
        f_tta = f1_score(ty, t_p.argmax(1), average="macro", zero_division=0)
        w = calibrate(v_p, vy, t_p.shape[1])
        f_cal = f1_score(ty, (t_p * w).argmax(1), average="macro", zero_division=0)
        rows.append((f"{name}+TTA8", f_tta))
        rows.append((f"{name}+TTA8+calib", f_cal))
        print(f"{name}: TTA={f_tta:.4f}  TTA+calib={f_cal:.4f}  w={np.round(w,3).tolist()}")

    if len(entries) >= 2:
        names = [n for n, _ in entries]
        v = np.mean([store[n]["v"] for n in names], 0)
        t = np.mean([store[n]["t"] for n in names], 0)
        vy, ty = store[names[0]]["vy"], store[names[0]]["ty"]
        f_ens = f1_score(ty, t.argmax(1), average="macro", zero_division=0)
        w = calibrate(v, vy, t.shape[1])
        f_ens_c = f1_score(ty, (t * w).argmax(1), average="macro", zero_division=0)
        tag = "+".join(names)
        rows.append((f"ENS[{tag}]+TTA8", f_ens))
        rows.append((f"ENS[{tag}]+TTA8+calib", f_ens_c))
        print(f"ensemble: TTA={f_ens:.4f}  TTA+calib={f_ens_c:.4f}")

    with open(os.path.join(REPO, "experiments/results.csv"), "a", newline="") as f:
        w = csv.writer(f)
        for n, s in rows:
            w.writerow([datetime.now().strftime("%m-%d %H:%M"), "evalplus", n, 0, "ok",
                        "", round(s, 4), round(s, 4), "", "", "", "", "", "eval_plus"])
    with open("/home/login/data/cle/auto/shared/leaderboard.md", "a") as f:
        f.write(f"\n## eval_plus {datetime.now().strftime('%m-%d %H:%M')}\n")
        for n, s in sorted(rows, key=lambda r: -r[1]):
            f.write(f"- {n}: test macro-F1 = {s:.4f}\n")
    print("已写入 results.csv 与 shared/leaderboard.md")


if __name__ == "__main__":
    main()
