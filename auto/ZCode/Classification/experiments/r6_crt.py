#!/usr/bin/env python3
"""R6 实验 1: cRT 与 τ-norm(Kang et al., ICLR 2020; facebookresearch/classifier-balancing)—— 在 baseline 阶段 1
checkpoint(全量微调, 实例均衡采样 = 原协议)上的后处理, 忠实于官方 config/ImageNet_LT/cls_crt.yaml 与 tau_norm.py:
  cRT: 主干冻结(推理态), 分类器重新初始化(Linear 960→K), 类均衡采样(ClassAwareSampler 等价: 按 1/n_c 加权有放回),
       SGD momentum 0.9, wd 5e-4, lr 0.2 按批量线性缩放(512→32: 0.0125), cosine 退火 10 epoch, 原协议训练增广;
       每 epoch 在 val 上评 Macro-F1, 取最优(与本协议选点一致; 官方取末 epoch)。
  τ-norm: baseline 末层权重逐类除以 ‖w_j‖^τ, 偏置置 0(官方 dotproduct, 无偏置); τ ∈ linspace(0, 2, 21) 按 val Macro-F1 选。
输出 weights_cle_ac_v1_r6/{mbv3_crt,mbv3_taunorm}_seed<s>/... 并调用 evaluate.py 评 val/test, 结果行追加到 results CSV。
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torchvision import transforms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from cle_fair_common import build_model, build_val_transform  # noqa: E402

PY = sys.executable
DATA = os.path.join(REPO, "datasets/cle_ac_v1")
ROOT = os.path.join(REPO, "weights_cle_ac_v1_r6")
CLASSES = ["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def items(split):
    return [(os.path.join(DATA, split, c, f), i) for i, c in enumerate(CLASSES)
            for f in sorted(os.listdir(os.path.join(DATA, split, c)))]


class DS(torch.utils.data.Dataset):
    def __init__(self, it, tf):
        self.it, self.tf = it, tf

    def __len__(self):
        return len(self.it)

    def __getitem__(self, i):
        p, y = self.it[i]
        return self.tf(Image.open(p).convert("RGB")), y


def train_tf():   # 与 train_cle_fair 训练增广完全一致
    return transforms.Compose([transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(degrees=15),
                               transforms.RandomCrop(size=(224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


@torch.no_grad()
def predict(model, loader):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(model(x.cuda()).argmax(1).cpu())
    return torch.cat(out).numpy()


def save_and_eval(model, name, seed, note, results):
    d = os.path.join(ROOT, f"{name}_seed{seed}", name)
    os.makedirs(d, exist_ok=True)
    ckpt = os.path.join(d, "model_best.pth")
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt)
    json.dump({"model": name, "norm_mean": MEAN, "norm_std": STD, "norm_source": "imagenet", "posthoc": note},
              open(os.path.join(d, "run_meta.json"), "w"), indent=2, ensure_ascii=False)
    for sp in ("val", "test"):
        subprocess.run([PY, "evaluate.py", "--model", name, "--weights", ckpt, "--data-path", f"./datasets/cle_ac_v1/{sp}"],
                       cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    res = {}
    for f in sorted(glob.glob(os.path.join(d, "eval_*", "results.json")), reverse=True):
        r = json.load(open(f))
        res.setdefault(os.path.basename(r["test_dir"]), r)
    t, v = res["test"], res["val"]
    sen = [round(t["per_class"][c]["sensitivity"], 4) for c in CLASSES]
    row = {"ts": datetime.now().strftime("%m-%d %H:%M"), "batch": "MHposthoc", "model": name, "seed": seed, "status": "ok",
           "val_f1": round(v["macro_f1"], 4), "test_f1": round(t["macro_f1"], 4), "test_acc": round(t["accuracy"], 4),
           "test_auroc": round(t["auc_roc_macro"], 4), "sens_benign": sen[0], "sens_malignant": sen[1], "sens_normal": sen[2],
           "elapsed_min": "", "note": json.dumps(note, ensure_ascii=False)[:120]}
    with open(results, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=list(row)).writerow(row)
    print(f"[{name}] seed{seed}: val {row['val_f1']} test {row['test_f1']} sensM {row['sens_malignant']} | {note}", flush=True)


def crt(seed, results, epochs=10, bs=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    src = torch.load(os.path.join(ROOT, f"mobilenet_v3_large_seed{seed}", "mobilenet_v3_large", "model_best.pth"), map_location="cpu")
    m = build_model("mbv3_crt", 3)
    miss, unexp = m.load_state_dict({k: v for k, v in src.items() if k.startswith("features.")}, strict=False)
    assert not unexp and all(k.startswith("classifier.") for k in miss)
    m.features.requires_grad_(False)
    head = m.classifier[1]
    g = torch.Generator().manual_seed(seed)
    nn.init.kaiming_uniform_(head.weight, a=5 ** 0.5, generator=g)          # nn.Linear 默认初始化(重新初始化)
    bound = 1 / head.in_features ** 0.5
    nn.init.uniform_(head.bias, -bound, bound, generator=g)
    m = m.cuda()
    tr_items = items("train")
    y = np.array([t for _, t in tr_items])
    w = 1.0 / np.bincount(y)[y]
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(y), replacement=True, generator=g)
    tl = torch.utils.data.DataLoader(DS(tr_items, train_tf()), batch_size=bs, sampler=sampler, num_workers=8, drop_last=False)
    vl = torch.utils.data.DataLoader(DS(items("val"), build_val_transform(224, MEAN, STD)), batch_size=64, num_workers=8)
    yv = np.array([t for _, t in items("val")])
    opt = torch.optim.SGD(head.parameters(), lr=0.2 * bs / 512, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tl), eta_min=0.0)
    best = (-1, None, -1)
    for ep in range(epochs):
        m.eval()                                   # 主干冻结且推理态; 头部无 BN/Dropout 训练语义需求外的状态
        m.classifier.train()
        for x, t in tl:
            with torch.amp.autocast("cuda"):
                loss = nn.functional.cross_entropy(m(x.cuda()), t.cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        f = f1_score(yv, predict(m, vl), average="macro")
        if f > best[0]:
            best = (f, {k: v.detach().clone() for k, v in m.classifier.state_dict().items()}, ep)
    m.classifier.load_state_dict(best[1])
    save_and_eval(m, "mbv3_crt", seed, {"method": "cRT", "best_epoch": best[2], "val_f1_select": round(best[0], 4),
                                         "lr": 0.2 * bs / 512, "epochs": epochs}, results)


@torch.no_grad()
def taunorm(seed, results):
    src = torch.load(os.path.join(ROOT, f"mobilenet_v3_large_seed{seed}", "mobilenet_v3_large", "model_best.pth"), map_location="cpu")
    m = build_model("mbv3_taunorm", 3)
    m.load_state_dict(src)
    m = m.cuda().eval()
    W = m.classifier[3].weight.detach().clone()
    vl = torch.utils.data.DataLoader(DS(items("val"), build_val_transform(224, MEAN, STD)), batch_size=64, num_workers=8)
    yv = np.array([t for _, t in items("val")])
    feats = []
    for x, _ in vl:
        h = m.avgpool(m.features(x.cuda())).flatten(1)
        feats.append(m.classifier[:3](h))           # 1280 维倒数第二层(Dropout 推理态为恒等)
    H = torch.cat(feats)
    best = (-1, None)
    for tau in np.linspace(0, 2, 21):
        Wt = W / W.norm(dim=1, keepdim=True).pow(float(tau))
        f = f1_score(yv, (H @ Wt.T).argmax(1).cpu().numpy(), average="macro")
        if f > best[0] + 1e-12:
            best = (f, float(tau))
    m.classifier[3].weight.copy_(W / W.norm(dim=1, keepdim=True).pow(best[1]))
    m.classifier[3].bias.zero_()
    save_and_eval(m, "mbv3_taunorm", seed, {"method": "tau-norm", "tau": round(best[1], 2), "val_f1_select": round(best[0], 4)}, results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--results", default=os.path.join(REPO, "experiments", "results_cle_ac_v1_r6.csv"))
    a = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for s in a.seeds:
        t0 = time.time()
        taunorm(s, a.results)
        crt(s, a.results)
        print(f"seed{s} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
