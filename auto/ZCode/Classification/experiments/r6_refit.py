#!/usr/bin/env python3
"""R6 阶段 2: 解耦凸证据头重拟合 (Decoupled Convex Evidence head, DCE)

阶段 1 checkpoint(主干按原协议端到端微调, 选点不变) -> 冻结主干, 在多尺度 GAP 证据上以强 L2 多项逻辑回归拟合
分类头; 正则强度 C 仅在 val 上按 Macro-F1 选择(并列取更强正则), 全程不读取 test。
预处理与 evaluate.py 完全一致: Resize(int(224*1.143)) -> CenterCrop(224) -> Normalize(阶段 1 run_meta 的 mean/std)。
输出: <dst-root>/<dst-model>_seed<seed>/<dst-model>/{model_best.pth, run_meta.json(含 refit 审计)}
      --eval: 再调用 evaluate.py 评 val/test, 并按 runner 同格式追加结果行到 --results。
用法: python experiments/r6_refit.py --src-model mobilenet_v3_large --dst-model mbv3_dce4s --seeds 42 43 44 \
        [--src-root weights_cle_ac_v1_r6] [--dst-root weights_cle_ac_v1_r6] [--data datasets/cle_ac_v1] --eval
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torchvision import transforms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from cle_fair_common import build_model  # noqa: E402

PY = sys.executable
CGRID = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(root, classes):
    paths, labels = [], []
    for i, c in enumerate(classes):
        for fn in sorted(os.listdir(os.path.join(root, c))):
            if fn.lower().endswith(EXTS):
                paths.append(os.path.join(root, c, fn))
                labels.append(i)
    return paths, np.array(labels)


class _DS(torch.utils.data.Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def pooled_features(model, paths, tf):
    model.eval()
    dl = torch.utils.data.DataLoader(_DS(paths, tf), batch_size=64, num_workers=8)
    out = []
    for x in dl:
        p = model.forward_pooled(x.cuda())
        out.append(torch.cat([p[t] for t in model.classifier.taps], 1).float().cpu())
    return torch.cat(out).numpy()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def refit_one(a, seed, classes):
    if a.src_ckpt:                               # 直接给定 checkpoint 路径(如纯 ImageNet pth → LP 臂)
        src_dir = os.path.join(REPO, a.dst_root, f"{a.dst_model}_seed{seed}", a.dst_model)
        src_ckpt = a.src_ckpt
    else:
        src_dir = os.path.join(REPO, a.src_root, f"{a.src_model}_seed{seed}", a.src_model)
        src_ckpt = os.path.join(src_dir, "model_best.pth")
    meta = ({"norm_mean": [0.485, 0.456, 0.406], "norm_std": [0.229, 0.224, 0.225], "norm_source": "imagenet"}
            if a.src_ckpt else json.load(open(os.path.join(src_dir, "run_meta.json"))))
    tf = transforms.Compose([transforms.Resize(int(224 * 1.143)), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize(meta["norm_mean"], meta["norm_std"])])
    model = build_model(a.dst_model, len(classes))
    sd = torch.load(src_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and "state_dict" not in sd and not any(k.startswith("features") for k in sd):
        sd = sd["model"]
    if hasattr(model, "load_stage1"):            # 通用主干(epic_generic): 原主干键 -> backbone.*, 丢弃原头部
        model.load_stage1(sd)
    else:
        feat_sd = {k: v for k, v in sd.items() if k.startswith("features.")}
        missing, unexpected = model.load_state_dict(feat_sd, strict=False)
        assert not unexpected and all(k.startswith("classifier.") for k in missing), (missing, unexpected)
    model.cuda()
    Xtr = pooled_features(model, list_images(os.path.join(REPO, a.data, "train"), classes)[0], tf)
    ytr = list_images(os.path.join(REPO, a.data, "train"), classes)[1]
    vpaths, yva = list_images(os.path.join(REPO, a.data, "val"), classes)
    Xva = pooled_features(model, vpaths, tf)
    mu = Xtr.mean(0)
    sigma = Xtr.std(0)
    sigma[sigma < 1e-8] = 1.0                   # 同 sklearn StandardScaler: 零方差维不缩放
    Ztr, Zva = (Xtr - mu) / sigma, (Xva - mu) / sigma
    grid, best = [], None
    for C in CGRID:                             # 升序: 并列时保留更强正则(更小 C)
        clf = LogisticRegression(C=C, max_iter=5000).fit(Ztr, ytr)
        f = float(f1_score(yva, clf.predict(Zva), average="macro"))
        grid.append({"C": C, "val_f1": round(f, 4)})
        if best is None or f > best[0]:
            best = (f, C, clf)
    clf = best[2]
    with torch.no_grad():
        model.classifier.mu.copy_(torch.from_numpy(mu))
        model.classifier.sigma.copy_(torch.from_numpy(sigma))
        model.classifier.fc.weight.copy_(torch.from_numpy(clf.coef_).float())
        model.classifier.fc.bias.copy_(torch.from_numpy(clf.intercept_).float())
    dst_dir = os.path.join(REPO, a.dst_root, f"{a.dst_model}_seed{seed}", a.dst_model)
    os.makedirs(dst_dir, exist_ok=True)
    dst_ckpt = os.path.join(dst_dir, "model_best.pth")
    torch.save(model.cpu().state_dict(), dst_ckpt)
    meta = dict({"norm_mean": [0.485, 0.456, 0.406], "norm_std": [0.229, 0.224, 0.225], "norm_source": "imagenet"}) if a.src_ckpt else dict(meta)
    meta.update({"model": a.dst_model, "stage": "R6 DCE stage-2 refit",
                 "refit": {"src_checkpoint": src_ckpt, "src_sha256": sha256(src_ckpt), "taps": list(model.classifier.taps),
                           "C_grid": grid, "C_selected": best[1], "val_macro_f1": round(best[0], 4),
                           "selection": "val Macro-F1 (ties -> stronger L2); test never read",
                           "n_train": int(len(ytr)), "n_val": int(len(yva)), "feature_dim": int(Xtr.shape[1])}})
    json.dump(meta, open(os.path.join(dst_dir, "run_meta.json"), "w"), indent=2, ensure_ascii=False)
    return dst_ckpt, best


def read_metric(dst_dir, which):
    for f in sorted(glob.glob(os.path.join(dst_dir, "eval_*", "results.json")), reverse=True):  # 最新优先(同目录重拟合时旧 eval 仍在)
        r = json.load(open(f))
        if ("val" in os.path.basename(r.get("test_dir", ""))) == (which == "val"):
            per = r.get("per_class", {})
            sen = [round(per[c]["sensitivity"], 4) for c in per]
            return r, sen
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", required=True)
    ap.add_argument("--src-ckpt", default="", help="直接指定源 checkpoint(如纯 ImageNet pth); 缺省按 src_root/src_model_seed<seed>")
    ap.add_argument("--dst-model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--src-root", default="weights_cle_ac_v1_r6")
    ap.add_argument("--dst-root", default="weights_cle_ac_v1_r6")
    ap.add_argument("--data", default="datasets/cle_ac_v1")
    ap.add_argument("--results", default="experiments/results_cle_ac_v1_r6.csv")
    ap.add_argument("--batch", default="refit")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    classes = sorted(d for d in os.listdir(os.path.join(REPO, a.data, "train"))
                     if os.path.isdir(os.path.join(REPO, a.data, "train", d)))
    env_cls = os.environ.get("RUN_CLASSES", "").split()
    is_cle = classes == ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]
    for seed in a.seeds:
        t0 = time.time()
        ckpt, best = refit_one(a, seed, classes)
        print(f"[refit] {a.dst_model} seed{seed}: C={best[1]} val_f1={best[0]:.4f} -> {ckpt}", flush=True)
        if not a.eval:
            continue
        dst_dir = os.path.dirname(ckpt)
        for sp in ("val", "test"):
            cmd = [PY, "evaluate.py", "--model", a.dst_model, "--weights", ckpt, "--data-path", f"./{a.data}/{sp}"]
            if env_cls:
                cmd += ["--class-names"] + env_cls
            subprocess.run(cmd, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        rv, _ = read_metric(dst_dir, "val")
        rt, sen = read_metric(dst_dir, "test")
        row = {"ts": datetime.now().strftime("%m-%d %H:%M"), "batch": a.batch, "model": a.dst_model, "seed": seed,
               "status": "ok", "val_f1": round(rv["macro_f1"], 4), "test_f1": round(rt["macro_f1"], 4),
               "test_acc": round(rt["accuracy"], 4), "test_auroc": round(rt.get("auc_roc_macro", float("nan")), 4),
               # 逐类敏感度列与 runner 同口径: 仅 CLE 三类填写, 其余域为 nan(完整逐类指标见 results.json)
               "sens_benign": sen[0] if is_cle else "nan", "sens_malignant": sen[1] if is_cle else "nan",
               "sens_normal": sen[2] if is_cle else "nan",
               "elapsed_min": round((time.time() - t0) / 60, 1), "note": f"refit<-{a.src_model} C={best[1]}"}
        path = os.path.join(REPO, a.results)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[eval] {a.dst_model} seed{seed}: val={row['val_f1']} test={row['test_f1']} sensM={row['sens_malignant']}", flush=True)


if __name__ == "__main__":
    main()
