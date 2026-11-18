# -*- coding: utf-8 -*-
"""
train_dual_fair.py — 双流融合训练 (classifyML/bow_fusion 版)

把 train_cle_fair.py 的「公平训练协议」与 ../../BoW/train_dual_fixed.py 的「LBP-BoW 双流分支」
融合在一起。classifyML 原始文件 (train_cle_fair.py / utils.py / cle_fair_common.py 等) 未被修改。

继承自 train_cle_fair.py 的协议:
  - 骨干通过 --model 选择 (cle_fair_common.MODEL_REGISTRY)，默认 fast_t0
  - 最优 checkpoint / 早停按 val MACRO-F1 选择
  - 保存 run_meta.json (含归一化、骨干、BoW 参数) 供评估复用同一预处理
  - 富金色进度条 (tqdm.rich, 宽度 = 终端 × 0.618)，本地 train_one_epoch
  - label_smoothing 支持；AdamW + CosineAnnealingLR + AMP

继承自 train_dual_fixed.py 的 BoW 分支:
  - --bow-K / --bow-scales / --align-bow / --bow-cache-dir
  - 预提取 LBP-BoW 特征 (align_bow=False 默认) 或实时对齐裁剪区提取 (align_bow=True)
  - 模型 forward 为 (images, bow_feat) 双输入

归一化: 用户加载预训练权重 → 直接用 ImageNet mean/std，不做 mean/std 自动计算
        (与 train_cle_fair.py 的 `if args.weights` 分支一致)；可用 --norm-mean/--norm-std 覆盖。

用法 (从 classifyML 根目录运行):
  python bow_fusion/train_dual_fair.py --model fast_t0 --num_classes 3 --img_size 224 \
      --weights <backbone_pretrained.pth> \
      --data-path-train /path/cle/train --data-path-val /path/cle/val \
      --save-dir ./bow_fusion/weights --epochs 300
"""

import os
import sys
import csv
import json
import time
import shutil
import random
import hashlib
import argparse
import warnings
from datetime import datetime, timedelta

import torch
import numpy as np
import torch.optim as optim
from torchvision import transforms
from tqdm.rich import tqdm as rich_tqdm
from rich.console import Console

# ── 确保 classifyML 根目录在 sys.path，以便 import cle_fair_common / utils ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from my_dataset import MyDataSet  # noqa: F401  (保留与原项目一致的命名空间)
from utils import read_split_train_data, read_split_val_data
from cle_fair_common import build_val_transform

from bow_extractor import extract_and_cache_bow, BOWFeatureCache
from my_dataset_bow import MyDataSetBOW
from dual_stream_model import dual_stream as create_model, ALL_BACKBONES

from sklearn.metrics import f1_score, accuracy_score


# ── 进度条样式：与 train_cle_fair.py / evaluate_2class.py 的 rich 金色进度条一致 ──
try:
    from tqdm import TqdmExperimentalWarning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
except Exception:
    pass
_TQDM_CONSOLE = Console(width=int(shutil.get_terminal_size().columns * 0.618))
_TQDM_OPTIONS = {'console': _TQDM_CONSOLE}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] seed = {seed} (set)")


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    scaler=None, label_smoothing=0.0):
    """单 epoch 训练 (三元组 images/bow_feats/labels)。rich 金色进度条，CE/AMP/AdamW。"""
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    use_amp = scaler is not None and device.type == "cuda"
    pbar = rich_tqdm(data_loader,
                     desc=f"train epoch {epoch} loss: 0.000, acc: 0.000",
                     options=_TQDM_OPTIONS)
    for step, data in enumerate(pbar):
        images, bow_feats, labels = data
        sample_num += images.shape[0]

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred = model(images.to(device), bow_feats.to(device))
            loss = loss_function(pred, labels.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        scaler.scale(loss).backward() if use_amp else loss.backward()
        scaler.step(optimizer) if use_amp else optimizer.step()
        scaler.update() if use_amp else None
        optimizer.zero_grad()
        accu_loss += loss.detach()

        pbar.set_description("train epoch {} loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate_val_macro_f1(model, data_loader, device, label_smoothing=0.0):
    """验证：返回 (val_loss, val_acc, val_macro_f1)。选择指标为 macro_f1。"""
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss, n_batches = 0.0, 0
    all_pred, all_true = [], []
    use_cuda = (device.type == "cuda")
    for images, bow_feats, labels in data_loader:
        images = images.to(device)
        bow_feats = bow_feats.to(device)
        labels = labels.to(device)
        with torch.amp.autocast('cuda', enabled=use_cuda):
            logits = model(images, bow_feats)
            loss = loss_fn(logits, labels)
        total_loss += loss.item()
        n_batches += 1
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_true.append(labels.cpu().numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    val_loss = total_loss / max(1, n_batches)
    val_acc = float(accuracy_score(y_true, y_pred))
    val_macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    return val_loss, val_acc, val_macro_f1


def _load_backbone_weights(model, weights_path, device):
    """把 --weights 载入双流模型。

    兼容两种权重:
      (a) 单骨干预训练权重 (key 形如 patch_embed.* / conv1.* / blocks.* + head/fc/classifier)：
          跳过 head/fc/classifier，其余 key 重映射为 backbone.{key} 后 strict=False 载入。
      (b) 既有双流模型权重 (含 backbone.* / semantic_proj.* / texture_mlp.* / classifier.*)：
          跳过 classifier (允许 num_classes 变化)，直接 strict=False 载入。
    """
    assert os.path.exists(weights_path), f"weights not found: {weights_path}"
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    # 兼容 {model: state_dict} 形式
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    own_keys = set(model.state_dict().keys())
    fused_prefixes = ("backbone.", "semantic_proj.", "texture_mlp.",
                      "cross_attn.", "norm_q.", "norm_kv.", "classifier.")

    if any(k.startswith(fused_prefixes) for k in sd.keys()):
        # (b) 双流权重：跳过 classifier
        sd2 = {k: v for k, v in sd.items()
               if k in own_keys and not k.startswith("classifier.")}
        miss = model.load_state_dict(sd2, strict=False)
        print(f"[weights] dual-stream checkpoint | loaded {len(sd2)} tensors | "
              f"missing={len(miss.missing_keys)}")
        return

    # (a) 单骨干权重：重映射为 backbone.* 前缀，跳过分类头
    head_top = ("head", "fc", "classifier")
    remapped, skipped = {}, []
    for k, v in sd.items():
        top = k.split('.')[0]
        if top in head_top:
            skipped.append(k)
            continue
        nk = f"backbone.{k}"
        if nk in own_keys and model.state_dict()[nk].shape == v.shape:
            remapped[nk] = v
        else:
            skipped.append(k)
    miss = model.load_state_dict(remapped, strict=False)
    print(f"[weights] backbone checkpoint | loaded {len(remapped)} tensors | "
          f"skipped {len(skipped)} (head/shape-mismatch) | "
          f"missing={len(miss.missing_keys)}")


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 解析 BoW 参数 ──
    scales = [int(s) for s in args.bow_scales.split(',')]
    bow_dim = len(scales) * args.bow_K

    # ── run dir (auto-suffix) ──
    model_name = f"dual_{args.model}"
    os.makedirs(args.save_dir, exist_ok=True)
    base_dir = os.path.join(args.save_dir, model_name)
    if not os.path.exists(base_dir):
        run_dir = base_dir
    else:
        idx = 1
        while os.path.exists(f"{base_dir}-{idx}"):
            idx += 1
        run_dir = f"{base_dir}-{idx}"
    os.makedirs(run_dir, exist_ok=True)

    save_path = os.path.join(run_dir, args.save_name)
    csv_path = os.path.join(run_dir, args.csv_name)
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 4])

    train_images_path, train_images_label = read_split_train_data(args.data_path_train)
    val_images_path, val_images_label = read_split_val_data(args.data_path_val)
    img_size = args.img_size

    # ── 归一化：ImageNet (用户加载预训练权重) / 可被 --norm-mean/--norm-std 覆盖 ──
    if args.norm_mean is not None and args.norm_std is not None:
        ds_mean = list(args.norm_mean)
        ds_std = list(args.norm_std)
        norm_source = "cli"
    else:
        ds_mean, ds_std = IMAGENET_MEAN, IMAGENET_STD
        norm_source = "imagenet"
    print(f"[norm] source={norm_source} MEAN={ds_mean} STD={ds_std}")

    # ── 提取 / 缓存 BoW 特征 ──
    scale_str = '_'.join(map(str, scales))
    train_hash = hashlib.md5(args.data_path_train.encode()).hexdigest()[:8]
    val_hash = hashlib.md5(args.data_path_val.encode()).hexdigest()[:8]
    train_cache = os.path.join(args.bow_cache_dir,
                               f'train_K{args.bow_K}_s{scale_str}_{train_hash}.pkl')
    val_cache = os.path.join(args.bow_cache_dir,
                             f'val_K{args.bow_K}_s{scale_str}_{val_hash}.pkl')

    print("[BoW] 提取 LBP-BoW 特征")
    if not args.align_bow:
        extract_and_cache_bow(args.data_path_train, train_cache, scales=scales, K=args.bow_K)
        extract_and_cache_bow(args.data_path_val, val_cache, scales=scales, K=args.bow_K)
        bow_cache_train = BOWFeatureCache(train_cache, bow_dim=bow_dim)
        bow_cache_val = BOWFeatureCache(val_cache, bow_dim=bow_dim)
    else:
        print("  align_bow=True: 跳过缓存，训练时实时提取裁剪区域 BoW 特征")
        bow_cache_train = bow_cache_val = None

    # ── 变换 (沿用 train_cle_fair.py) ──
    data_transform = {
        "train": transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.RandomRotation(degrees=15),
                                     transforms.RandomCrop(size=(img_size, img_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(ds_mean, ds_std)]),
        "val":   build_val_transform(img_size, ds_mean, ds_std)}

    train_dataset = MyDataSetBOW(images_path=train_images_path,
                                 images_class=train_images_label,
                                 transform=data_transform["train"],
                                 bow_cache=bow_cache_train,
                                 align_bow=args.align_bow,
                                 scales=scales, K=args.bow_K)
    val_dataset = MyDataSetBOW(images_path=val_images_path,
                               images_class=val_images_label,
                               transform=data_transform["val"],
                               bow_cache=bow_cache_val,
                               align_bow=args.align_bow,
                               scales=scales, K=args.bow_K)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, pin_memory=True, num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, pin_memory=True, num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(backbone_name=args.model, num_classes=args.num_classes,
                         bow_dim=bow_dim).to(device)
    print(f"[model] dual_stream backbone={args.model} feat_dim={model.feat_dim} "
          f"bow_dim={bow_dim} d_model={model.d_model}")

    if args.weights:
        _load_backbone_weights(model, args.weights, device)

    # ── 可选：冻结骨干，只训融合头 ──
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.backbone.parameters())
        print(f"[freeze] backbone frozen ({frozen:,} params), trainable={trainable:,}")

    # ── run_meta.json ──
    run_meta = {
        "model": model_name,
        "backbone": args.model,
        "num_classes": args.num_classes,
        "img_size": img_size,
        "seed": args.seed,
        "norm_mean": ds_mean,
        "norm_std": ds_std,
        "norm_source": norm_source,
        "selection_metric": "val_macro_f1",
        "loss": (f"label_smoothed_ce (smoothing={args.label_smoothing})"
                 if args.label_smoothing > 0 else "unweighted_cross_entropy"),
        "bow_dim": bow_dim,
        "bow_K": args.bow_K,
        "bow_scales": scales,
        "align_bow": args.align_bow,
        "freeze_backbone": args.freeze_backbone,
        "data_path_train": os.path.abspath(args.data_path_train),
        "data_path_val": os.path.abspath(args.data_path_val),
        "weights": os.path.abspath(save_path),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5e-2)

    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_macro_f1', 'lr'])

    print(f"[train] backbone={args.model} num_classes={args.num_classes} img={img_size} "
          f"bs={args.batch_size} epochs={args.epochs} | CE (label_smoothing={args.label_smoothing}) | "
          f"select=val_macro_f1 | BoW(K={args.bow_K}, scales={scales}, align_bow={args.align_bow})")
    print(f"[out]   run_dir={run_dir}")

    best_macro_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer,
                                                data_loader=train_loader, device=device,
                                                epoch=epoch, scaler=scaler,
                                                label_smoothing=args.label_smoothing)

        val_loss, val_acc, val_macro_f1 = evaluate_val_macro_f1(
            model, val_loader, device, label_smoothing=args.label_smoothing)

        scheduler.step()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch,
                                    f'{train_loss:.6f}', f'{train_acc:.6f}',
                                    f'{val_loss:.6f}', f'{val_acc:.6f}',
                                    f'{val_macro_f1:.6f}', f'{current_lr:.8f}'])

        is_best = val_macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        flag = " *" if is_best else ""
        print(f"Epoch {epoch}: train_loss={train_loss:.3f}, train_acc={train_acc:.3f}, "
              f"val_loss={val_loss:.3f}, val_acc={val_acc:.3f}, val_macroF1={val_macro_f1:.4f}, "
              f"lr={current_lr:.6f}, best_macroF1={best_macro_f1:.4f} (epoch {best_epoch}){flag}")

        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(f"\n[early stop] epoch {epoch}: val_macro_f1 not improved for {args.early_stop} epochs")
            break

    total_time_str = str(timedelta(seconds=int(time.time() - start_time)))
    loss_desc = (f"CrossEntropy (label_smoothing={args.label_smoothing})"
                 if args.label_smoothing > 0 else "unweighted CrossEntropy")
    print()
    print("=" * 62)
    print(f"{'Training Summary (dual-stream fair)':^62}")
    print("=" * 62)
    print(f"  {'Model:':<22}{model_name}")
    print(f"  {'Backbone:':<22}{args.model}")
    print(f"  {'Best val macro-F1:':<22}{best_macro_f1:.4f}  (epoch {best_epoch})")
    print(f"  {'Selection:':<22}val macro-F1")
    print(f"  {'Loss:':<22}{loss_desc}")
    print(f"  {'BoW:':<22}K={args.bow_K}, scales={scales}, align_bow={args.align_bow}")
    print(f"  {'Norm:':<22}{norm_source}")
    print(f"  {'Training Time:':<22}{total_time_str}")
    print(f"  {'Weights:':<22}{save_path}")
    print(f"  {'Run meta:':<22}{os.path.join(run_dir, 'run_meta.json')}")
    print("=" * 62)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed (42/43/44 for repeats)')
    parser.add_argument('--model', type=str, default='fast_t0',
                        help='backbone name; one of: ' + ", ".join(ALL_BACKBONES))
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='label smoothing for CrossEntropy loss (0 = off)')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--data-path-train', type=str, default="/path/to/cle/train",
                        help='patient-level train split directory')
    parser.add_argument('--data-path-val', type=str, default="/path/to/cle/val",
                        help='patient-level val split directory')
    parser.add_argument('--weights', type=str, default='',
                        help='backbone pretrained weights (or a prior dual-stream checkpoint)')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='freeze the CNN backbone, train only the fusion head')
    parser.add_argument('--device', default='cuda:0', help='device id (e.g. cuda:0 or cpu)')
    parser.add_argument('--save-dir', type=str, default='./bow_fusion/weights')
    parser.add_argument('--save-name', type=str, default='model_best.pth')
    parser.add_argument('--csv-name', type=str, default='train_log.csv')
    parser.add_argument('--early-stop', type=int, default=30,
                        help='early stopping patience on val_macro_f1 (0 = disabled)')
    parser.add_argument('--norm-mean', type=float, nargs=3, default=None,
                        help='override normalization mean (default: ImageNet)')
    parser.add_argument('--norm-std', type=float, nargs=3, default=None,
                        help='override normalization std (default: ImageNet)')
    # BoW 参数
    parser.add_argument('--bow-K', type=int, default=256, help='BoW codebook size (bins per scale)')
    parser.add_argument('--bow-scales', type=str, default='16,32,64',
                        help='multi-scale patch sizes (comma-separated)')
    parser.add_argument('--bow-cache-dir', type=str, default='./bow_fusion/bow_cache',
                        help='directory for BoW feature cache files')
    parser.add_argument('--align-bow', action='store_true',
                        help='extract BoW on the same crop region as the CNN (slower)')
    opt = parser.parse_args()
    main(opt)
