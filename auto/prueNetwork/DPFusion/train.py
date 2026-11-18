import os
import csv
import json
import hashlib
import argparse
import time
import warnings
import random

import torch
import numpy as np
from thop import profile, clever_format
import torch.optim as optim
from datetime import datetime, timedelta
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchinfo import summary
from my_dataset import MyDataSet
from utils import read_split_train_data, train_one_epoch, evaluate, read_split_val_data


# ── 设置随机数种子（确保可复现性）──

def set_seed(seed=42):
    """设置随机数种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保卷积算法的确定性（会略微降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[随机数种子] seed = {seed} (已设置)")


# ── 自动计算数据集 Mean/Std（带 fingerprint 缓存）──

_NORM_CACHE_FILE = "dataset_norm_cache.json"


def _norm_transform_desc(img_size):
    resize_size = int(img_size * 1.143)
    return f"Resize({resize_size}) + CenterCrop({img_size}) + ToTensor()"


def _norm_cache_key(data_path, img_size):
    h = hashlib.md5(os.path.abspath(data_path).encode()).hexdigest()[:8]
    return f"{h}_img{img_size}"


def _dataset_fingerprint(image_paths, data_path):
    abs_root = os.path.abspath(data_path)
    parts = []
    for p in sorted(image_paths):
        rel = os.path.relpath(p, abs_root)
        sz = os.path.getsize(p)
        mt = os.path.getmtime(p)
        parts.append(f"{rel}|{sz}|{mt}")
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_norm_cache(cache_dir):
    path = os.path.join(cache_dir, _NORM_CACHE_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        warnings.warn(f"缓存文件 {path} 读取失败 ({e})，将重新计算 mean/std")
        return {}


def _save_norm_cache(cache_dir, cache):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, _NORM_CACHE_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def compute_dataset_mean_std(image_paths, img_size, batch_size=64, num_workers=4):
    if not image_paths:
        raise ValueError("image_paths 为空，无法计算 mean/std")

    resize_size = int(img_size * 1.143)
    temp_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    class _TempDataset(Dataset):
        def __init__(self, paths, tfm):
            self.paths = paths
            self.tfm = tfm
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            with Image.open(self.paths[idx]) as img:
                return self.tfm(img.convert("RGB"))

    loader = DataLoader(_TempDataset(image_paths, temp_transform),
                        batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=True)

    ch_sum = torch.zeros(3, dtype=torch.float64)
    ch_sum_sq = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0
    for batch in loader:
        batch = batch.to(torch.float64)
        b, c, h, w = batch.shape
        n_pixels += b * h * w
        ch_sum += batch.sum(dim=[0, 2, 3])
        ch_sum_sq += (batch ** 2).sum(dim=[0, 2, 3])

    mean = ch_sum / n_pixels
    var = ch_sum_sq / n_pixels - mean ** 2
    var = torch.clamp(var, min=0.0)
    std = torch.clamp(torch.sqrt(var), min=1e-12)

    return mean.tolist(), std.tolist()


def get_or_compute_mean_std(data_path, image_paths, img_size,
                            cache_dir, batch_size=64, num_workers=4,
                            force_recompute=False):
    key = _norm_cache_key(data_path, img_size)
    transform_desc = _norm_transform_desc(img_size)
    num_images = len(image_paths)
    fingerprint = _dataset_fingerprint(image_paths, data_path)

    if not force_recompute:
        cache = _load_norm_cache(cache_dir)
        entry = cache.get(key)
        if entry is not None:
            valid = (
                entry.get("dataset_fingerprint") == fingerprint
                and entry.get("num_images") == num_images
                and entry.get("img_size") == img_size
                and entry.get("transform") == transform_desc
            )
            if valid:
                return entry["mean"], entry["std"], "hit"

    t0 = time.time()
    mean, std = compute_dataset_mean_std(image_paths, img_size, batch_size, num_workers)
    elapsed = time.time() - t0

    cache = _load_norm_cache(cache_dir)
    cache[key] = {
        "mean": mean,
        "std": std,
        "dataset_fingerprint": fingerprint,
        "num_images": num_images,
        "img_size": img_size,
        "resize_size": int(img_size * 1.143),
        "crop_size": img_size,
        "transform": transform_desc,
        "data_path": os.path.abspath(data_path),
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(elapsed, 1),
    }
    _save_norm_cache(cache_dir, cache)

    status = "forced recompute" if force_recompute else "miss"
    return mean, std, status


# from models.RepViT import repvit_m1 as create_model
try:
    from models.FasterNet import fast_t0 as create_model
except ImportError:  # standalone DPFusion package ships only Convnext/cle_convnext
    create_model = None
# from models.VanilaNet import vanillanet_5 as create_model
# from models.Mobilevit import mobile_vit_xx_small as create_model
# from models.Resnet import resnet34 as create_model
# from models.Deit import deit_base_patch16_224 as create_model
# from models.Pvt import pvt_tiny as create_model
# from models.DenseNet import densenet121 as create_model


def main(args):
    # ── 设置随机数种子 ──
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 初始化（计算在前，统一在模型架构后输出）──
    model_name = create_model.__name__
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

    # ── 自动计算 Mean/Std（仅训练集，带 fingerprint 缓存）──
    norm_cache_dir = args.norm_cache_dir
    transform_desc = _norm_transform_desc(img_size)

    print("[0/6] 自动获取训练集 Mean / Std ...")
    print(f"  Mean/Std source: training set only")
    print(f"  Train path:      {args.data_path_train}")
    print(f"  Train images:    {len(train_images_path)}")
    print(f"  img_size:        {img_size}")
    print(f"  Norm transform:  {transform_desc}")

    ds_mean, ds_std, cache_status = get_or_compute_mean_std(
        args.data_path_train, train_images_path, img_size,
        cache_dir=norm_cache_dir,
        batch_size=args.batch_size, num_workers=nw,
        force_recompute=args.recompute_norm)

    status_map = {
        "hit": "hit (缓存命中)",
        "miss": "miss (缓存未命中，已计算并写入)",
        "forced recompute": "forced recompute (--recompute-norm，已重算并覆盖)",
    }
    print(f"  Cache:           {status_map.get(cache_status, cache_status)}")
    print(f"  MEAN = [{ds_mean[0]:.4f}, {ds_mean[1]:.4f}, {ds_mean[2]:.4f}]")
    print(f"  STD  = [{ds_std[0]:.4f}, {ds_std[1]:.4f}, {ds_std[2]:.4f}]")
    print()

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(ds_mean, ds_std)]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(ds_mean, ds_std)])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)
    myinput = torch.zeros((1, 3, img_size, img_size)).to(device)
    macs_raw, params_raw = profile(model, inputs=(myinput,), verbose=False)
    flops_raw = 2 * macs_raw
    macs_str, params_str = clever_format([macs_raw, params_raw], "%.2f")
    flops_str, _ = clever_format([flops_raw, 0], "%.2f")

    # ── 输出模型架构 ──
    summary(model, inputs=(myinput,))

    # ── [1/6] 配置训练环境 ──
    print(f"[1/6] 配置训练环境")
    print(f"  设备:       {device}")
    print(f"  工作线程:   {nw}")
    print(f"  输出目录:   {run_dir}")
    print(f"  权重文件:   {save_path}")
    print(f"  CSV 日志:   {csv_path}")
    print()

    # ── [2/6] 数据集信息 ──
    print(f"[2/6] 数据集信息")
    print(f"  训练集:     {len(train_images_path)} 张  ({args.data_path_train})")
    print(f"  验证集:     {len(val_images_path)} 张  ({args.data_path_val})")
    print(f"  批大小:     {args.batch_size}")
    print()

    # ── [3/6] 模型复杂度 ──
    print(f"[3/6] 模型复杂度: {model_name}")
    print(f"  {'MACs:':<12}{macs_str}")
    print(f"  {'FLOPs:':<12}{flops_str}  (1 MAC = 2 FLOPs)")
    print(f"  {'Params:':<12}{params_str}")
    print()

    # ── [4/6] 加载预训练权重（可选）──
    has_weights = args.weights != ""
    has_freeze = args.freeze_layers
    if has_weights:
        print(f"[4/6] 加载预训练权重")
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device, weights_only=True)["model"]
        skipped = [k for k in weights_dict if "head" in k]
        for k in skipped:
            del weights_dict[k]
        result = model.load_state_dict(weights_dict, strict=False)
        print(f"  权重文件:   {args.weights}")
        if skipped:
            print(f"  跳过键:     {', '.join(skipped)}")
        if result.missing_keys:
            print(f"  缺失键:     {', '.join(result.missing_keys)}")
    elif has_freeze:
        print(f"[4/6] 冻结层（无预训练权重）")
    else:
        print(f"[4/6] 从零开始训练（无预训练权重）")

    trainable_names = []
    if has_freeze:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                trainable_names.append(name)
        if trainable_names:
            print(f"  可训练参数: {', '.join(trainable_names)}")
    print()

    # ── [5/6] 配置优化器 ──
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    early_stop_str = f"{args.early_stop} epochs" if args.early_stop > 0 else "关闭"
    print(f"[5/6] 配置优化器")
    print(f"  优化器:     AdamW  (lr={args.lr}, weight_decay=0.05)")
    print(f"  调度器:     CosineAnnealingLR  (T_max={args.epochs})")
    print(f"  早停:       {early_stop_str}")
    print()

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

    best_acc = 0.
    best_epoch = 1
    start_time = time.time()
    patience_counter = 0

    # ── [6/6] 开始训练 ──
    print(f"[6/6] 开始训练  (共 {args.epochs} epochs)")
    print()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                scaler=scaler)

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        scheduler.step()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch,
                             f'{train_loss:.6f}', f'{train_acc:.6f}',
                             f'{val_loss:.6f}', f'{val_acc:.6f}',
                             f'{current_lr:.8f}'])

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        best_flag = " *" if is_best else ""
        print(f"Epoch {epoch}: train_loss={train_loss:.3f}, train_acc={train_acc:.3f}, "
              f"val_loss={val_loss:.3f}, val_acc={val_acc:.3f}, lr={current_lr:.6f}, "
              f"best_acc={best_acc:.3f} (epoch {best_epoch}){best_flag}")

        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(f"\n  Early stopping at epoch {epoch}, no improvement for {args.early_stop} epochs")
            break

    # ── 训练结束摘要 ──
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))

    print()
    print("=" * 59)
    print(f"{'Training Summary':^59}")
    print("=" * 59)
    print(f"  {'Model:':<16}{create_model.__name__}")
    print(f"  {'MACs:':<16}{macs_str}  (thop profile)")
    print(f"  {'FLOPs:':<16}{flops_str}  (1 MAC = 2 FLOPs)")
    print(f"  {'Params:':<16}{params_str}")
    print(f"  {'─' * 55}")
    print(f"  {'Best Accuracy:':<16}{best_acc:.3f}  (epoch {best_epoch})")
    print(f"  {'Training Time:':<16}{total_time_str}")
    print(f"  {'Device:':<16}{device}")
    print(f"  {'─' * 55}")
    print(f"  {'Weights:':<16}{save_path}")
    print(f"  {'CSV Log:':<16}{csv_path}")
    print("=" * 59)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='随机数种子，用于确保实验可复现')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--data-path-train', type=str,
                        default="/home/u2318483076/filtered_abcd/train")
    parser.add_argument('--data-path-val', type=str,
                        default="/home/u2318483076/filtered_abcd/val")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', action='store_true')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--save-dir', type=str, default='./weights',
                        help='directory to save model weights and training log')
    parser.add_argument('--save-name', type=str, default='resnet34_cle.pth',
                        help='filename for saved model weights')
    parser.add_argument('--csv-name', type=str, default='resnet34_cle.csv',
                        help='filename for training log')
    parser.add_argument('--early-stop', type=int, default=0,
                        help='early stopping patience (0 to disable)')
    # Mean/Std 缓存参数
    parser.add_argument('--recompute-norm', action='store_true',
                        help='强制重新计算 mean/std，忽略已有缓存')
    parser.add_argument('--norm-cache-dir', type=str, default='./norm_cache',
                        help='mean/std 缓存目录')

    opt = parser.parse_args()
    main(opt)
