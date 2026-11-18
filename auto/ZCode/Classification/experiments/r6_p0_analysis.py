#!/usr/bin/env python3
"""R6 P0 扩展分析 — MobileNetV3-L 基座创新动机的数据验证 (2026-11-08)

A. 历史 mbv3 (10-26 批, seeds 42-44) test 错误结构: 读 predictions.csv —— 混淆对 / 亚型 / 跨 seed 一致性
B. 逐尺度线性探针: 冻结特征 (ImageNet 预训练 + 微调 s42-44) 在 f3/f6/f12/f15/f16 的 GAP
   -> 标准化 + 多项逻辑回归 (C 在 val 上选) -> test Macro-F1 / 恶性敏感度 / B-vs-M AUC, 及尺度拼接增量
C. 空间离散度: 同层 [GAP] vs [GAP, 空间std] 的探针增量 —— 局部异质性是否被 GAP 丢弃
D. 结构可解码性: 输入 Sobel 边缘密度 / 局部强度 std (FOV 内, 腐蚀去圆形视野边) 从 f12/f16 逐位置特征的
   岭回归 R^2 (val 拟合 -> test 评估), 预训练 vs 微调 —— CE 微调是否丢弃腺体边界几何
输出: experiments/r6_p0_analysis.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")   # joblib 并行探针, 防 BLAS 线程超订
import csv
import glob
import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from PIL import Image
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score, recall_score, roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from models.MobileNetV3 import mobilenet_v3_large  # noqa: E402

DATA = os.path.join(REPO, "datasets/cle_ac_v1")
OUT_JSON = os.path.join(REPO, "experiments", "r6_p0_analysis.json")
CLASSES = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]
NAME_RE = re.compile(r"^(train|val|test)_(.+?)_(S_[0-9A-Za-z]+)\.[^.]+$")
PTH = "/home/login/data/cle/sunfuxiang/classifyML/pth/mobilenet_v3_large-8738ca79.pth"
FT_DIR = os.path.join(REPO, "weights_cle_ac_v1", "mobilenet_v3_large_seed{}", "mobilenet_v3_large")
SEEDS = (42, 43, 44)
LAYERS = {"f3": 3, "f6": 6, "f12": 12, "f15": 15, "f16": 16}   # /4 24ch, /8 40ch, /16 112ch, /32 160ch, /32 960ch
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)        # mbv3 run_meta: norm_source=imagenet
PROBE_CFGS = {
    "f3": [("gap", "f3")], "f6": [("gap", "f6")], "f12": [("gap", "f12")],
    "f15": [("gap", "f15")], "f16": [("gap", "f16")],
    "f16+f12": [("gap", "f16"), ("gap", "f12")],
    "f16+f12+f6": [("gap", "f16"), ("gap", "f12"), ("gap", "f6")],
    "f16+f12+f6+f3": [("gap", "f16"), ("gap", "f12"), ("gap", "f6"), ("gap", "f3")],
    "f12[gap+std]": [("gap", "f12"), ("sd", "f12")],
    "f16[gap+std]": [("gap", "f16"), ("sd", "f16")],
    "f16+f12[gap+std]": [("gap", "f16"), ("sd", "f16"), ("gap", "f12"), ("sd", "f12")],
}


def subtype_of(path):
    m = NAME_RE.match(os.path.basename(path))
    return m.group(2) if m else "?"


# ───────────────────────── A. 历史错误结构 ─────────────────────────
def find_test_eval(run_dir):
    for f in sorted(glob.glob(os.path.join(run_dir, "eval_*", "results.json"))):
        r = json.load(open(f))
        if "test" in r.get("test_dir", ""):
            return os.path.dirname(f), r
    raise FileNotFoundError(f"no test eval under {run_dir}")


def part_a():
    out = {"per_seed": {}}
    wrong, truth = defaultdict(list), {}
    for s in SEEDS:
        d, r = find_test_eval(FT_DIR.format(s))
        rows = list(csv.DictReader(open(os.path.join(d, "predictions.csv"))))
        cm = np.zeros((3, 3), int)
        sub_err, sub_n = Counter(), Counter()
        yt, pm = [], []
        for row in rows:
            t, p = CLASSES.index(row["true_label"]), CLASSES.index(row["pred_label"])
            cm[t, p] += 1
            key = f"{row['true_label'][:1]}:{subtype_of(row['image_path'])}"
            sub_n[key] += 1
            truth[row["image_path"]] = key
            yt.append(t)
            pm.append([float(row[f"prob_{c}"]) for c in CLASSES])
            if t != p:
                sub_err[key] += 1
                wrong[row["image_path"]].append(row["pred_label"][:1])
        yt, pm = np.array(yt), np.array(pm)
        bm = yt != 2
        out["per_seed"][s] = {
            "macro_f1": round(r["macro_f1"], 4), "kappa": round(r["kappa"], 4),
            "sens_m": round(float(cm[1, 1] / cm[1].sum()), 4),
            "auc_bm": round(float(roc_auc_score(yt[bm] == 1, pm[bm, 1] / (pm[bm, 0] + pm[bm, 1] + 1e-12))), 4),
            "cm": cm.tolist(),
            "errors_by_subtype": {k: f"{sub_err[k]}/{sub_n[k]}" for k in sorted(sub_n)},
        }
    cons, core = Counter(), Counter()
    for pth, preds in wrong.items():
        cons[len(preds)] += 1
        if len(preds) == len(SEEDS):
            core[f"{truth[pth]}->{Counter(preds).most_common(1)[0][0]}"] += 1
    out["wrong_in_k_of_3_seeds"] = {k: cons[k] for k in sorted(cons)}
    out["hard_core_3of3"] = dict(core.most_common())
    out["n_test"] = len(truth)
    return out


# ───────────────────────── 数据缓存 ─────────────────────────
def list_split(sp):
    paths, labels = [], []
    for i, c in enumerate(CLASSES):
        for fn in sorted(os.listdir(os.path.join(DATA, sp, c))):
            paths.append(os.path.join(DATA, sp, c, fn))
            labels.append(i)
    return paths, np.array(labels)


class _DS(torch.utils.data.Dataset):
    # 与 evaluate.py build_transform 一致: Resize(int(224*1.143)=256, 短边) -> CenterCrop(224); 归一化留到 GPU 上做
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def load_tensors(paths):
    dl = torch.utils.data.DataLoader(_DS(paths), batch_size=64, num_workers=8)
    return torch.cat(list(dl))


SOBEL = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8


def structure_targets(x):
    """x (B,3,224,224)∈[0,1] -> {f12: (edge, local_std, fov_frac) @14², f16: 同 @7²}."""
    g = x.mean(1, keepdim=True)
    fov = (F.avg_pool2d(g, 15, 1, 7, count_include_pad=False) > 0.03).float()
    fov = 1 - F.max_pool2d(1 - fov, 15, 1, 7)          # 腐蚀 7px: 去掉圆形视野边界的人工强边
    k = torch.stack([SOBEL, SOBEL.t()]).unsqueeze(1)
    mag = F.conv2d(g, k, padding=1).pow(2).sum(1, keepdim=True).sqrt() * fov
    out = {}
    for name, s in (("f12", 16), ("f16", 32)):
        frac = F.avg_pool2d(fov, s)
        den = frac.clamp_min(1e-6)
        m1, m2 = F.avg_pool2d(g * fov, s) / den, F.avg_pool2d(g * g * fov, s) / den
        out[name] = (F.avg_pool2d(mag, s) / den, (m2 - m1 * m1).clamp_min(0).sqrt(), frac)
    return out


# ───────────────────────── B/C/D. 特征 ─────────────────────────
@torch.no_grad()
def extract(model, X, keep_maps):
    acts = {}
    hooks = [model.features[i].register_forward_hook(lambda m, inp, o, k=k: acts.__setitem__(k, o))
             for k, i in LAYERS.items()]
    gap, sd, maps = defaultdict(list), defaultdict(list), defaultdict(list)
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
    for i in range(0, len(X), 128):
        model((X[i:i + 128].cuda() - mean) / std)
        for k in LAYERS:
            a = acts[k].float()
            gap[k].append(a.mean(dim=(2, 3)).cpu())
            sd[k].append(a.std(dim=(2, 3)).cpu())
        if keep_maps:
            for k in ("f12", "f16"):
                maps[k].append(acts[k].half().cpu())
    for h in hooks:
        h.remove()
    cat = lambda d: {k: torch.cat(v).numpy() for k, v in d.items()}  # noqa: E731
    return cat(gap), cat(sd), cat(maps)


def probe(Xs, ys, cfg):
    (Xtr, Xva, Xte), (ytr, yva, yte) = Xs, ys
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best = None
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr)
        f = f1_score(yva, clf.predict(Xva), average="macro")
        if best is None or f > best[0]:
            best = (f, C, clf)
    p = best[2].predict_proba(Xte)
    pred = p.argmax(1)
    bm = yte != 2
    return {"cfg": cfg, "dim": Xtr.shape[1], "C": best[1], "val_f1": round(best[0], 4),
            "test_f1": round(f1_score(yte, pred, average="macro"), 4),
            "sens_m": round(float(recall_score(yte, pred, labels=[1], average=None)[0]), 4),
            "auc_bm": round(float(roc_auc_score(yte[bm] == 1, p[bm, 1] / (p[bm, 0] + p[bm, 1] + 1e-12))), 4)}


def decodability(m_va, m_te, t_va, t_te, fr_va, fr_te):
    """逐位置特征 -> 结构目标 的岭回归 R^2 (仅完全落在腐蚀 FOV 内的位置)."""
    def flat(m, t, fr):
        Xf = m.transpose(0, 2, 3, 1).reshape(-1, m.shape[1]).astype(np.float32)
        keep = fr.reshape(-1) > 0.95
        return Xf[keep], t.reshape(-1)[keep]
    Xa, ya = flat(m_va, t_va, fr_va)
    Xb, yb = flat(m_te, t_te, fr_te)
    sc = StandardScaler().fit(Xa)
    r = RidgeCV(alphas=(0.1, 1, 10, 100, 1e3, 1e4)).fit(sc.transform(Xa), ya)
    return {"r2_test": round(float(r2_score(yb, r.predict(sc.transform(Xb)))), 4),
            "alpha": float(r.alpha_), "n_fit": int(len(ya)), "n_test": int(len(yb))}


def main():
    torch.backends.cudnn.benchmark = False
    res = {"A_errors": part_a()}
    print(json.dumps(res["A_errors"], indent=1, ensure_ascii=False), flush=True)

    splits = {sp: list_split(sp) for sp in ("train", "val", "test")}
    X = {sp: load_tensors(splits[sp][0]) for sp in splits}
    y = {sp: splits[sp][1] for sp in splits}
    tg = {sp: structure_targets(X[sp]) for sp in ("val", "test")}
    print("[data] cached", {sp: tuple(X[sp].shape) for sp in X}, flush=True)

    feats, res["D_decodability"] = {}, {}
    for name, seed in [("imnet", None)] + [(f"ft{s}", s) for s in SEEDS]:
        m = mobilenet_v3_large(num_classes=1000 if seed is None else 3)
        ck = PTH if seed is None else os.path.join(FT_DIR.format(seed), "model_best.pth")
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.eval().cuda()
        out = {sp: extract(m, X[sp], keep_maps=(sp != "train")) for sp in X}
        feats[name] = {sp: (out[sp][0], out[sp][1]) for sp in out}
        dec = {}
        for k in ("f12", "f16"):
            for ti, tname in ((0, "edge"), (1, "local_std")):
                dec[f"{k}:{tname}"] = decodability(
                    out["val"][2][k], out["test"][2][k],
                    tg["val"][k][ti].numpy(), tg["test"][k][ti].numpy(),
                    tg["val"][k][2].numpy(), tg["test"][k][2].numpy())
        res["D_decodability"][name] = dec
        print(f"[D] {name}", json.dumps(dec), flush=True)
        del m, out
        torch.cuda.empty_cache()

    jobs = []
    for name in feats:
        for cfg, parts in PROBE_CFGS.items():
            Xs = tuple(np.concatenate([feats[name][sp][0 if kind == "gap" else 1][layer] for kind, layer in parts], 1)
                       for sp in ("train", "val", "test"))
            jobs.append((name, cfg, Xs))
    outs = Parallel(n_jobs=12)(delayed(probe)(Xs, (y["train"], y["val"], y["test"]), cfg) for _, cfg, Xs in jobs)
    res["BC_probes"] = defaultdict(dict)
    for (name, cfg, _), o in zip(jobs, outs):
        res["BC_probes"][name][cfg] = o

    # 摘要: 微调 3 seeds 均值
    print("\n%-20s %6s | %-24s | %-24s" % ("probe", "dim", "imnet f1/sensM/aucBM", "ft(3s) f1/sensM/aucBM"))
    summary = {}
    for cfg in PROBE_CFGS:
        im = res["BC_probes"]["imnet"][cfg]
        ft = [res["BC_probes"][f"ft{s}"][cfg] for s in SEEDS]
        agg = {k: round(float(np.mean([o[k] for o in ft])), 4) for k in ("test_f1", "sens_m", "auc_bm")}
        summary[cfg] = {"imnet": {k: im[k] for k in ("test_f1", "sens_m", "auc_bm")}, "ft_mean": agg}
        print("%-20s %6d | %.4f/%.4f/%.4f | %.4f/%.4f/%.4f" % (cfg, im["dim"], im["test_f1"], im["sens_m"],
              im["auc_bm"], agg["test_f1"], agg["sens_m"], agg["auc_bm"]))
    res["BC_summary"] = summary
    net = res["A_errors"]["per_seed"]
    print("network itself (ft mean): f1=%.4f sensM=%.4f aucBM=%.4f" % tuple(
        np.mean([net[s][k] for s in SEEDS]) for k in ("macro_f1", "sens_m", "auc_bm")))
    json.dump(res, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    print("saved ->", OUT_JSON)


if __name__ == "__main__":
    main()
