# -*- coding: utf-8 -*-
"""
dataset_quality_grading.py
基于患者级 K 折 Out-of-Fold 预测 + 多轮随机种子重复评估的
CLE 内镜图像训练集样本质量分级系统（5 级：A/B/C/D/E）

科学逻辑:
  1. 基于论文 s41598-022-18707-6 的聚类假设：位于特征空间低密度区域
     （分类边界附近）的样本通常特征模糊或标注存疑。
  2. 基于集成学习的不确定性估计：多个独立模型对同一样本的预测一致性
     越高，该样本越可能位于类别特征的核心区域。
  3. 每张 train 图像只获得来自"未见过该患者"的模型的预测（OOF），
     确保评估无偏。

使用方式:
  python dataset_quality_grading.py \
      --data-path-train /home/u2318483067/twoClass/train \
      --csv-path /home/u2318483067/csv/train_processed.csv \
      --model-module models.Resnet \
      --model-function resnet34 \
      --num-rounds 3 --num-folds 5 --epochs 300

  # 快速测试
  python dataset_quality_grading.py --epochs 2 --num-rounds 1
"""

# ═══════════════════════════════════════════════════════════════════════
# 与 train1.py 的一致性对照
# ═══════════════════════════════════════════════════════════════════════
#
#  一致:
#    - optimizer:   AdamW, lr=0.0001, weight_decay=5E-2
#    - scheduler:   无（train1.py 中 CosineAnnealing/ExponentialLR 均已注释）
#    - transforms:  train: RandomResizedCrop(224) + RandomHorizontalFlip
#                             + ToTensor + Normalize(ImageNet)
#                   val:   Resize(256) + CenterCrop(224) + ToTensor + Normalize
#    - batch_size:  32
#    - loss:        CrossEntropyLoss（无 label smoothing）
#    - img_size:    224
#    - mixed precision: 未使用（train1.py 也未使用）
#
#  有意差异:
#    - checkpoint:  train1.py 保存 best model state_dict；
#                  本脚本仅保存 OOF 预测结果 JSON，不保存模型权重。
#                  原因：评估样本质量而非产出部署模型。
#    - DataLoader:  本脚本增加了 generator + worker_init_fn 确保可复现，
#                  train1.py 未做此控制。
#    - num_classes: 本脚本默认 2（CLE二分类），train1.py 默认 3。
# ═══════════════════════════════════════════════════════════════════════

import os
import sys
import csv
import json
import copy
import random
import argparse
import importlib
import time
from datetime import timedelta
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score

# ── 复用项目现有模块 ──
from my_dataset import MyDataSet
from utils import train_one_epoch


# ═══════════════════════════════════════════════════════════════════════
# 1. 全局工具函数
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """固定所有随机源，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int):
    """DataLoader worker 的种子初始化，确保多 worker 可复现"""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练集样本质量分级系统（患者级 K-Fold OOF 预测）")

    # ── 数据路径 ──
    parser.add_argument('--data-path-train', type=str,
                        default='/home/u2318483078/cledata_3class/train',
                        help='训练集图片根目录（含 normal/ 和 abnormal/ 子目录）')
    parser.add_argument('--csv-path', type=str,
                        default='/home/u2318483078/csv/train_processed.csv',
                        help='train_processed.csv 路径，用于 sample_id→patient_id 映射')

    # ── 模型 ──
    parser.add_argument('--model-module', type=str, default='models.Resnet',
                        help='模型模块路径，如 models.Resnet / models.FasterNet')
    parser.add_argument('--model-function', type=str, default='resnet34',
                        help='模型创建函数名，如 resnet34 / resnet50 / fast_t0')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--weights', type=str, default='',
                        help='预训练权重路径（可选）')
    parser.add_argument('--freeze-layers', action='store_true', default=False,
                        help='冻结除 head 外的所有层')

    # ── 训练超参（与 train1.py 保持一致）──
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--device', default='cuda:0')

    # ── K-Fold 与重复评估 ──
    parser.add_argument('--num-folds', type=int, default=5,
                        help='患者级 K 折数')
    parser.add_argument('--num-rounds', type=int, default=3,
                        help='重复评估轮次（每轮不同随机种子划分折）')
    parser.add_argument('--base-seed', type=int, default=42,
                        help='基础随机种子，第 r 轮种子 = base_seed + r')

    # ── Early Stopping ──
    parser.add_argument('--use-early-stop', action='store_true', default=False,
                        help='启用内层患者级早停机制（不使用外层OOF fold）')
    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='早停耐心轮数（inner-val指标连续多少epoch未改善则停止）')
    parser.add_argument('--early-stop-min-delta', type=float, default=1e-4,
                        help='指标最小改善阈值')
    parser.add_argument('--inner-val-ratio', type=float, default=0.2,
                        help='从当前训练fold内划分出的患者级early-stop验证集比例')
    parser.add_argument('--early-stop-monitor', type=str, default='macro_f1',
                        choices=['loss', 'acc', 'macro_f1'],
                        help='早停监控指标（基于inner-val）')
    parser.add_argument('--early-stop-min-epochs', type=int, default=10,
                        help='最低训练轮数: 前 N 个 epoch 不触发早停'
                             '(仍正常记录 best)，避免极早期随机 checkpoint')

    # ── 输出 ──
    parser.add_argument('--output-dir', type=str,
                        default='./outputs_3class/quality_grading',
                        help='所有中间结果与最终输出的根目录')
    parser.add_argument('--output-csv', type=str,
                        default='./outputs_3class/train_sample_grading.csv',
                        help='最终样本分级 CSV 路径')
    parser.add_argument('--patient-summary-csv', type=str,
                        default='./outputs_3class/train_patient_grading_summary.csv',
                        help='患者级汇总 CSV 路径')

    # ── 五级分级阈值 ──
    parser.add_argument('--grade_a_acc', type=float, default=0.95,
                        help='Grade A 最低 pred_correct_rate')
    parser.add_argument('--grade_a_conf', type=float, default=0.95,
                        help='Grade A 最低 mean_confidence')
    parser.add_argument('--grade_b_acc', type=float, default=0.85,
                        help='Grade B 最低 pred_correct_rate')
    parser.add_argument('--grade_b_conf', type=float, default=0.85,
                        help='Grade B 最低 mean_confidence')
    parser.add_argument('--grade_d_acc', type=float, default=0.60,
                        help='低于此 pred_correct_rate → Grade D')
    parser.add_argument('--grade_d_conf', type=float, default=0.60,
                        help='低于此 mean_confidence → Grade D')
    parser.add_argument('--grade_e_acc', type=float, default=0.30,
                        help='低于此 pred_correct_rate → Grade E')
    parser.add_argument('--grade_e_conf', type=float, default=0.30,
                        help='低于此 mean_confidence → Grade E')

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# 2. 数据读取与患者 ID 识别
# ═══════════════════════════════════════════════════════════════════════

def load_csv_mapping(csv_path: str) -> dict:
    """
    读取 train_processed.csv，返回 {sample_id: patient_id} 映射字典。
    只保留 split == 'train' 的行。
    CSV 使用逗号分隔符。
    """
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('split', '').strip() == 'train':
                sample_id = row['sample_id'].strip()
                patient_id = row['leakage_group_id'].strip()
                mapping[sample_id] = patient_id
    return mapping


def extract_sample_id(image_path: str) -> str:
    """
    从图片文件名中提取 sample_id。
    文件名格式: train_normal_S_002cb6015eed.png
                train_atrophy_S_0001e5e611a4.png
                train_intestinal_metaplasia_S_xxx.png

    提取逻辑: 找到 'S_' 开头的部分直到扩展名之前。
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]  # 去掉 .png
    # basename 形如: train_normal_S_002cb6015eed
    # 找到 S_ 的位置
    idx = basename.find('S_')
    if idx == -1:
        raise ValueError(
            f"无法从文件名中提取 sample_id: {image_path}\n"
            f"期望文件名中包含 'S_' 前缀的 sample ID。")
    return basename[idx:]


def extract_patient_id(image_path: str, sample_to_patient: dict) -> str:
    """
    优先级 1: 通过 CSV 映射 sample_id → patient_id
    优先级 2: 如果 CSV 中无对应记录，尝试从文件名/路径中解析

    当前数据集使用 CSV 映射方式。
    """
    sample_id = extract_sample_id(image_path)
    if sample_id in sample_to_patient:
        return sample_to_patient[sample_id]

    # 备用方式：尝试从路径中提取（父目录或文件名前缀）
    # 当前数据集不需要此分支，但保留扩展性
    raise ValueError(
        f"无法识别患者 ID: {image_path}\n"
        f"提取到的 sample_id '{sample_id}' 在 CSV 中无对应记录。")


def scan_train_samples(train_root: str, sample_to_patient: dict):
    """
    扫描 train 目录，构建每张图片的信息记录。
    支持两种目录结构:
      1) 扁平: train_root/normal/*.png, train_root/abnormal/*.png
      2) 嵌套: train_root/normal/P001/*.png, train_root/normal/P001/sub/*.png

    使用 os.walk 递归收集所有支持的图片文件。

    返回:
        samples: list[dict]，每个 dict 包含:
            - image_path: 图片绝对路径
            - true_label: 类别索引 (abnormal=0, normal=1, 按字母序)
            - patient_id: 患者 ID
            - sample_id: 样本 ID
        class_indices: dict, 类别名→索引映射
    """
    supported = [".jpg", ".JPG", ".png", ".PNG", ".jpeg"]
    flower_class = sorted([
        cla for cla in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, cla))
    ])
    class_indices = {k: v for v, k in enumerate(flower_class)}
    # abnormal=0, normal=1

    samples = []
    for cla in flower_class:
        cla_path = os.path.join(train_root, cla)
        label = class_indices[cla]
        # 使用 os.walk 递归收集，兼容 class/ 和 class/patient/ 结构
        for dirpath, dirnames, filenames in os.walk(cla_path):
            dirnames.sort()  # 保证遍历顺序一致
            for fname in sorted(filenames):
                if os.path.splitext(fname)[-1] in supported:
                    full_path = os.path.join(dirpath, fname)
                    sample_id = extract_sample_id(full_path)
                    patient_id = extract_patient_id(full_path, sample_to_patient)
                    samples.append({
                        'image_path': full_path,
                        'true_label': label,
                        'patient_id': patient_id,
                        'sample_id': sample_id,
                    })

    print(f"[数据扫描] 共 {len(samples)} 张图片, "
          f"{len(set(s['patient_id'] for s in samples))} 名患者")
    print(f"  类别映射: {class_indices}")
    return samples, class_indices


# ═══════════════════════════════════════════════════════════════════════
# 3. 患者级 K 折划分
# ═══════════════════════════════════════════════════════════════════════

def build_patient_groups(samples):
    """
    按患者分组，并计算每个患者的多数类标签（用于分层抽样）。

    返回:
        patient_groups: {patient_id: [sample_dict, ...]}
        patient_majority_label: {patient_id: majority_class_int}
    """
    patient_groups = defaultdict(list)
    for s in samples:
        patient_groups[s['patient_id']].append(s)

    patient_majority_label = {}
    for pid, imgs in patient_groups.items():
        labels = [s['true_label'] for s in imgs]
        counter = Counter(labels)
        patient_majority_label[pid] = counter.most_common(1)[0][0]

    return dict(patient_groups), patient_majority_label


def build_patient_folds(patient_majority_label: dict,
                        n_splits: int = 5,
                        seed: int = 42):
    """
    患者级分层 K 折划分。

    方案说明:
      使用 StratifiedGroupKFold 而非 StratifiedKFold + patient majority label。
      在本场景中，每个 patient_id 天然是一个 group（同一患者的所有图像必须同折），
      同时我们希望按每个患者的"多数类标签"进行分层，使各折中 abnormal/normal 患者
      比例大致一致。

      StratifiedGroupKFold 是 sklearn 提供的专门处理"分组 + 分层"的方案，
      它在保证 group 不被拆分的前提下，尽量维持 y (stratify label) 的比例均衡。

      由于我们以患者为最小单位（每个患者一条记录传入 split），group 和 sample
      是 1:1 的，因此 StratifiedGroupKFold 的分组约束自动满足"同患者同折"。
      这比 "StratifiedKFold on patient majority label" 更加语义明确，
      且在少数患者标签分布不均匀时（如某折中多类患者偏多）有更好的平衡性。

    近似性与适用前提:
      - 分层标签使用的是患者的"多数类标签"，对于同时拥有 normal 和 abnormal
        图像的患者（约 60%），这是一种近似——患者内少数类图像的标签信息被忽略。
      - 这意味着分层只保证各折中"以 abnormal 为主的患者"与"以 normal 为主的
        患者"比例均衡，不保证各折图像级别的类别比例完全一致。
      - 对于当前数据集（213 名患者，中位数 23 张/患者），这种近似是合理的，
        因为患者级均衡是防止泄漏的首要目标。

    返回:
        folds: list of (train_patient_ids, val_patient_ids) 长度为 n_splits
    """
    from sklearn.model_selection import StratifiedGroupKFold

    patient_ids = sorted(patient_majority_label.keys())
    labels = [patient_majority_label[pid] for pid in patient_ids]
    X = np.arange(len(patient_ids))  # 用索引作为 X
    y = np.array(labels)
    groups = np.arange(len(patient_ids))  # 每个患者唯一 group

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                 random_state=seed)
    folds = []
    for train_idx, val_idx in sgkf.split(X, y, groups):
        train_pids = [patient_ids[i] for i in train_idx]
        val_pids = [patient_ids[i] for i in val_idx]
        folds.append((train_pids, val_pids))

    return folds


def split_inner_train_val_by_patient(train_pids, patient_majority,
                                      ratio=0.1, seed=42):
    """
    从外层 fold 的 train_pids 中按患者级切分出 inner-val。
    用于 early stopping，不触碰外层 OOF fold。

    方法学原则:
      - inner-val 仅用于指导训练过程（何时停止），不参与最终 OOF 评分。
      - 外层 OOF fold 上的预测仍然来自"从未见过该患者"的模型，无偏性不受影响。
      - 同一患者不会同时出现在 inner-train 和 inner-val 中。

    返回:
        inner_train_pids: list[str]
        inner_val_pids: list[str]
    """
    from sklearn.model_selection import train_test_split

    pids = sorted(train_pids)
    labels = [patient_majority[pid] for pid in pids]

    n_total = len(pids)
    n_val = max(1, int(n_total * ratio))  # 至少1名患者

    # 尝试分层划分
    try:
        inner_train_pids, inner_val_pids = train_test_split(
            pids, test_size=n_val,
            stratify=labels,
            random_state=seed,
            shuffle=True,
        )
    except ValueError:
        # 某类患者数太少无法分层（如某类仅1名），回退到随机划分
        print(f"    [inner-split] 患者类别数不均，"
              f"回退到非分层随机划分 (seed={seed})")
        inner_train_pids, inner_val_pids = train_test_split(
            pids, test_size=n_val,
            random_state=seed,
            shuffle=True,
        )

    return inner_train_pids, inner_val_pids


def flatten_patient_subset(patient_ids: list,
                           patient_groups: dict):
    """
    将患者 ID 列表展开为 (image_paths, image_labels)。
    """
    paths = []
    labels = []
    for pid in patient_ids:
        for s in patient_groups[pid]:
            paths.append(s['image_path'])
            labels.append(s['true_label'])
    return paths, labels


# ═══════════════════════════════════════════════════════════════════════
# 4. 模型创建（动态导入）
# ═══════════════════════════════════════════════════════════════════════

def create_model_from_args(args):
    """
    通过 importlib 动态导入模型，与 train1.py 中的导入模式一致。
    用法: --model-module models.Resnet --model-function resnet34
    """
    module = importlib.import_module(args.model_module)
    model_fn = getattr(module, args.model_function)
    model = model_fn(num_classes=args.num_classes)
    return model


# ═══════════════════════════════════════════════════════════════════════
# 5. 逐样本推理（OOF 预测）
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_oof_for_fold(model, data_loader, device):
    """
    对数据集进行逐样本推理，返回每张图片的详细预测结果。

    返回 list[dict]:
      - pred_label: 预测类别 (0/1)
      - pred_prob:  真实标签对应的概率（不是 max 概率）
      - confidence: softmax 最大概率（用于辅助分析）
      - loss:       基于真实标签的交叉熵损失
      - correct:    预测是否正确 (0/1)
      - softmax_all: 完整 softmax 输出 [p0, p1]
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    all_results = []

    for images, labels, paths in tqdm(data_loader, desc="  [OOF预测]",
                                       file=sys.stdout, leave=False):
        images_dev = images.to(device)
        labels_dev = labels.to(device)

        logits = model(images_dev)
        probs = torch.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)

        # 逐样本交叉熵损失
        losses = loss_fn(logits, labels_dev)

        # 真实标签对应的概率
        true_label_probs = probs[
            torch.arange(len(labels_dev)), labels_dev
        ]

        corrects = (pred_labels == labels_dev).long()

        for i in range(len(labels)):
            true_label = labels[i].item()
            softmax_vec = probs[i].cpu().tolist()
            result = {
                'image_path': paths[i],                   # 保留路径用于匹配
                'pred_label': pred_labels[i].item(),
                'pred_prob': true_label_probs[i].item(),  # 真实标签对应概率
                'confidence': probs[i].max().item(),       # 最大概率
                'loss': losses[i].item(),
                'correct': corrects[i].item(),
                'softmax_all': softmax_vec,
                'true_label': true_label,
            }
            all_results.append(result)

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# 6. 单折训练 + OOF 预测
# ═══════════════════════════════════════════════════════════════════════

def build_transforms(img_size: int):
    """构建训练和验证 transforms。

    使用 ImageNet 标准归一化(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    以匹配 ImageNet 预训练权重(ResNet/ViT 均在 ImageNet 归一化下预训练)，
    使预训练特征在微调时正常生效。
    """
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }
    return data_transform


@torch.no_grad()
def evaluate_one_epoch(model, data_loader, device):
    """
    单 epoch 验证评估，返回 (val_loss, val_acc, val_macro_f1)。
    用于 inner-val early stopping 监控，不影响外层 OOF 预测。
    复用 utils.py 中 evaluate 的计算逻辑，但以函数形式返回聚合指标。
    macro_f1 口径与 train_cle_fair.py 的 evaluate_val_macro_f1 一致:
    sklearn f1_score(average='macro', zero_division=0)。
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    all_pred, all_true = [], []

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_fn(pred, labels.to(device))
        accu_loss += loss
        all_pred.append(pred_classes.cpu().numpy())
        all_true.append(labels.cpu().numpy())

    val_loss = accu_loss.item() / (step + 1)
    val_acc = accu_num.item() / sample_num
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    val_macro_f1 = float(
        f1_score(y_true, y_pred, average='macro', zero_division=0))
    return val_loss, val_acc, val_macro_f1


def train_one_fold(train_paths, train_labels,
                   val_paths, val_labels,
                   val_samples, args, round_idx, fold_idx,
                   num_train_patients=0,
                   train_pids=None, patient_groups=None,
                   patient_majority=None):
    """
    训练一个模型并对验证折进行 OOF 预测。

    早停机制说明（当 args.use_early_stop == True 时生效）:
      - 从外层 fold 的 train_pids 中再按患者级划分出 inner-val（默认 10% 患者）
      - 使用 inner-val 的 loss/acc 作为 early stopping 监控指标
      - 外层 held-out fold (val_pids) 完全不参与 early stopping 决策
      - 这保证了外层 OOF 预测的无偏性：模型从未因 val_pids 的表现而调整
      - 早停的作用是减少无效训练和降低过训练风险，不是替代外层 OOF 评估

    返回:
        fold_results: list[dict]，每张 val 图片的预测结果 + 元信息
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_transform = build_transforms(args.img_size)
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 4])

    # ── Early Stopping: inner split ──
    inner_val_loader = None
    use_early_stop = args.use_early_stop

    if use_early_stop:
        if train_pids is None or patient_groups is None or patient_majority is None:
            print("  [警告] use_early_stop=True 但缺少患者信息，回退到不使用早停")
            use_early_stop = False
        elif len(train_pids) < 5:
            print(f"  [警告] 训练患者仅 {len(train_pids)} 名，"
                  f"无法稳定划分 inner-val，回退到不使用早停")
            use_early_stop = False

    if use_early_stop:
        inner_seed = args.base_seed + round_idx * 1000 + fold_idx * 10
        inner_train_pids, inner_val_pids = split_inner_train_val_by_patient(
            train_pids, patient_majority,
            ratio=args.inner_val_ratio, seed=inner_seed)

        # 用 inner split 的患者重新生成训练路径
        inner_train_paths, inner_train_labels = flatten_patient_subset(
            inner_train_pids, patient_groups)
        inner_val_paths, inner_val_labels = flatten_patient_subset(
            inner_val_pids, patient_groups)

        print(f"  [早停] R{round_idx}/F{fold_idx} inner split: "
              f"inner-train {len(inner_train_pids)} 患者 "
              f"({len(inner_train_paths)} 张), "
              f"inner-val {len(inner_val_pids)} 患者 "
              f"({len(inner_val_paths)} 张), "
              f"monitor={args.early_stop_monitor}, "
              f"patience={args.early_stop_patience}")

        # 覆盖 train_paths/labels 为 inner-train 部分
        train_paths = inner_train_paths
        train_labels = inner_train_labels

        # 构建 inner-val DataLoader（使用 MyDataSet，不需要路径保留）
        inner_val_dataset = MyDataSet(
            images_path=inner_val_paths,
            images_class=inner_val_labels,
            transform=data_transform["val"])
        inner_val_loader = DataLoader(
            inner_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=nw,
            collate_fn=inner_val_dataset.collate_fn,
            worker_init_fn=_worker_init_fn)

    # ── 构建 Dataset 与 DataLoader ──
    train_dataset = MyDataSet(images_path=train_paths,
                              images_class=train_labels,
                              transform=data_transform["train"])

    g = torch.Generator()
    g.manual_seed(args.base_seed + round_idx * 100 + fold_idx)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=nw,
                              collate_fn=train_dataset.collate_fn,
                              generator=g,
                              worker_init_fn=_worker_init_fn)

    val_loader_for_pred = DataLoader(
        _PathPreservingDataset(val_paths, val_labels,
                               transform=data_transform["val"]),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=_path_preserving_collate,
        worker_init_fn=_worker_init_fn
    )

    # ── 创建模型 ──
    model = create_model_from_args(args).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), \
            f"weights file: '{args.weights}' not exist."
        # weights_only=False 兼容 legacy tar 格式权重(如 torchvision 2017 版 resnet34)
        weights_dict = torch.load(args.weights, map_location='cpu',
                                  weights_only=False)
        if isinstance(weights_dict, dict) and 'model' in weights_dict:
            weights_dict = weights_dict['model']
        elif isinstance(weights_dict, dict) and 'state_dict' in weights_dict:
            weights_dict = weights_dict['state_dict']
        # 砍掉分类头:按顶层模块名精确匹配 head/fc/classifier,
        # 避免子串误伤(如 mlp.fc1)。删掉后这些键由新类别数随机初始化。
        head_prefixes = ('head', 'fc', 'classifier')
        removed = [k for k in list(weights_dict.keys())
                   if k.split('.')[0] in head_prefixes]
        for k in removed:
            del weights_dict[k]
        load_result = model.load_state_dict(weights_dict, strict=False)
        # 打印加载情况,便于核对:主干必须全部加载,只允许分类头 missing
        print(f"  [权重] 砍掉分类头键: {removed}")
        print(f"  [权重] missing_keys: {load_result.missing_keys}")
        print(f"  [权重] unexpected_keys: {load_result.unexpected_keys}")

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if name.split('.')[0] not in ('head', 'fc', 'classifier'):
                para.requires_grad_(False)

    # ── 优化器 ──
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    # ── Early Stopping 状态 ──
    best_model_state = None
    best_epoch = 0
    if use_early_stop:
        # loss: 越小越好，初始化为正无穷
        # acc / macro_f1: 越大越好，初始化为负无穷
        if args.early_stop_monitor == 'loss':
            best_metric = float('inf')
        else:
            best_metric = float('-inf')
        epochs_no_improve = 0

    # ── 日志 ──
    log_prefix = f"  [训练{'+早停' if use_early_stop else ''}] " \
                 f"R{round_idx}/F{fold_idx}:"
    print(f"{log_prefix} "
          f"训练 {len(train_paths)} 张 ({num_train_patients} 患者), "
          f"OOF验证 {len(val_paths)} 张")

    # ── 按当前折实际训练标签计算 balanced 类别权重 ──
    # 注意:此处 train_labels 已是"实际用于训练的 labels"
    #       (若启用 early-stop,inner split 已将其覆盖为 inner-train 部分)
    from collections import Counter as _Counter
    _cnt = _Counter(train_labels)
    _n_total = len(train_labels)
    _n_cls = args.num_classes
    _weights = []
    for _c in range(_n_cls):
        _cc = _cnt.get(_c, 0)
        # balanced: n_total / (n_classes * count_c); 某类为0时回退权重1.0
        _weights.append(_n_total / (_n_cls * _cc) if _cc > 0 else 1.0)
    class_weights = torch.tensor(_weights, dtype=torch.float32)
    print(f"  [类别权重] R{round_idx}/F{fold_idx} balanced weights="
          f"{[round(w, 4) for w in _weights]} (counts={dict(_cnt)})")

    # ── 训练循环 ──
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model, optimizer=optimizer,
            data_loader=train_loader,
            device=device, epoch=epoch,
            class_weights=class_weights)

        # ── Early Stopping 检查 ──
        if use_early_stop and inner_val_loader is not None:
            val_loss, val_acc, val_macro_f1 = evaluate_one_epoch(
                model, inner_val_loader, device)

            if args.early_stop_monitor == 'loss':
                current_metric = val_loss
                improved = current_metric < best_metric - args.early_stop_min_delta
            elif args.early_stop_monitor == 'acc':
                current_metric = val_acc
                improved = current_metric > best_metric + args.early_stop_min_delta
            else:  # macro_f1，越大越好，与 acc 同方向
                current_metric = val_macro_f1
                improved = current_metric > best_metric + args.early_stop_min_delta

            if improved:
                best_metric = current_metric
                best_epoch = epoch
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
                improved_flag = "*"
            else:
                epochs_no_improve += 1
                improved_flag = " "

            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1 or improved:
                print(f"    Epoch {epoch+1:3d}/{args.epochs} "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                      f"inner_val_loss={val_loss:.4f} inner_val_acc={val_acc:.4f} "
                      f"inner_val_macro_f1={val_macro_f1:.4f} "
                      f"{improved_flag}")

            # 最低训练轮数保护: 前 min_epochs 个 epoch 不触发早停(仍正常记录 best)
            if (epochs_no_improve >= args.early_stop_patience
                    and (epoch + 1) >= args.early_stop_min_epochs):
                print(f"    [早停] Epoch {epoch+1}: 连续 "
                      f"{args.early_stop_patience} epoch 无改善，停止训练 "
                      f"(best_epoch={best_epoch+1}, "
                      f"best_{args.early_stop_monitor}={best_metric:.6f})")
                break
        else:
            # 无早停：仅打印常规日志
            if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
                print(f"    Epoch {epoch+1}/{args.epochs} "
                      f"loss={train_loss:.4f} acc={train_acc:.4f}")

    # ── 恢复 best model（如有早停）──
    if use_early_stop and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"    [恢复] 使用 best_epoch={best_epoch+1} 的模型参数进行 OOF 预测")

    # ── OOF 预测 ──
    print(f"  [OOF预测] R{round_idx}/F{fold_idx}: "
          f"对 {len(val_paths)} 张图片进行推理...")
    raw_results = predict_oof_for_fold(model, val_loader_for_pred, device)

    # ── 组装完整结果（通过路径匹配预测与样本元信息）──
    sample_lookup = {s['image_path']: s for s in val_samples}
    fold_results = []
    for pred in raw_results:
        path = pred['image_path']
        if path not in sample_lookup:
            print(f"  [警告] 预测路径无法匹配到样本: {path}")
            continue
        sample = sample_lookup[path]
        fold_results.append({
            'image_path': sample['image_path'],
            'sample_id': sample['sample_id'],
            'patient_id': sample['patient_id'],
            'true_label': sample['true_label'],
            'round': round_idx,
            'fold': fold_idx,
            'pred_label': pred['pred_label'],
            'pred_prob': pred['pred_prob'],
            'confidence': pred['confidence'],
            'loss': pred['loss'],
            'correct': pred['correct'],
            'softmax_all': pred['softmax_all'],
        })

    return fold_results


# ── 辅助：保留路径信息的 Dataset 和 collate ──

class _PathPreservingDataset(torch.utils.data.Dataset):
    """与 MyDataSet 类似，但在 __getitem__ 中额外返回路径"""

    def __init__(self, images_path, images_class, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        from PIL import Image
        img = Image.open(self.images_path[item]).convert('RGB')
        label = self.images_class[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, self.images_path[item]


def _path_preserving_collate(batch):
    """自定义 collate：保留路径字符串"""
    images, labels, paths = tuple(zip(*batch))
    images = torch.stack(images, dim=0)
    labels = torch.as_tensor(labels)
    return images, labels, paths


# ═══════════════════════════════════════════════════════════════════════
# 7. 断点续训
# ═══════════════════════════════════════════════════════════════════════

def get_fold_dir(output_dir, model_name, round_idx, fold_idx):
    """获取单折输出目录"""
    d = os.path.join(output_dir, model_name,
                     f'repeat_{round_idx}', f'fold_{fold_idx}')
    os.makedirs(d, exist_ok=True)
    return d


def get_fold_result_path(fold_dir):
    return os.path.join(fold_dir, 'oof_predictions.json')


def check_fold_completed(fold_dir, expected_count):
    """
    检查某折是否已完成。
    条件: oof_predictions.json 存在、可解析、结果数量正确。
    """
    result_path = get_fold_result_path(fold_dir)
    if not os.path.exists(result_path):
        return False
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        if 'results' not in data:
            return False
        if len(data['results']) != expected_count:
            return False
        # 抽查第一条记录是否有必需字段
        required_keys = {'image_path', 'pred_label', 'pred_prob',
                         'confidence', 'loss', 'correct', 'true_label'}
        if data['results'] and not required_keys.issubset(data['results'][0].keys()):
            return False
        return True
    except (json.JSONDecodeError, KeyError, IndexError):
        return False


def save_fold_results(fold_dir, results, config_dict):
    """保存单折预测结果"""
    result_path = get_fold_result_path(fold_dir)
    data = {
        'config': config_dict,
        'num_results': len(results),
        'results': results,
    }
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [保存] {result_path} ({len(results)} 条记录)")


def load_fold_results(fold_dir):
    """加载单折预测结果"""
    result_path = get_fold_result_path(fold_dir)
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']


# ═══════════════════════════════════════════════════════════════════════
# 8. 汇总与分级
# ═══════════════════════════════════════════════════════════════════════

def aggregate_predictions(output_dir, model_name,
                          num_rounds, num_folds,
                          all_image_paths):
    """
    汇总所有折的 OOF 预测结果，按 image_path 聚合。

    返回:
        aggregated: dict, key=image_path, value=dict 包含:
            - sample_id, patient_id, true_label
            - pred_correct_rate: 预测正确比例
            - mean_confidence: 真实标签对应概率的均值
            - mean_entropy: 平均预测熵
            - avg_loss: 平均交叉熵损失
            - num_predictions: 预测次数
            - std_confidence: 置信度标准差
            - all_confidences: 所有置信度列表
            - all_correct_flags: 所有正确标志列表
            - all_losses: 所有损失列表
            - prediction_sources: 预测来源列表 [{"round":r,"fold":f}, ...]
    """
    # 用一个有序结构收集所有路径的结果
    image_results = defaultdict(lambda: {
        'sample_id': None,
        'patient_id': None,
        'true_label': None,
        'pred_labels': [],
        'pred_probs': [],      # 真实标签对应概率
        'confidences': [],      # 最大概率
        'losses': [],
        'corrects': [],
        'softmax_alls': [],
        'sources': [],          # 预测来源 (round, fold)
    })

    loaded_count = 0
    for round_idx in range(num_rounds):
        for fold_idx in range(num_folds):
            fold_dir = get_fold_dir(output_dir, model_name,
                                    round_idx, fold_idx)
            result_path = get_fold_result_path(fold_dir)
            if not os.path.exists(result_path):
                print(f"  [错误] 缺少结果文件: {result_path}")
                continue

            results = load_fold_results(fold_dir)
            loaded_count += 1
            for r in results:
                path = r['image_path']
                entry = image_results[path]
                entry['sample_id'] = r['sample_id']
                entry['patient_id'] = r['patient_id']
                entry['true_label'] = r['true_label']
                entry['pred_labels'].append(r['pred_label'])
                entry['pred_probs'].append(r['pred_prob'])
                entry['confidences'].append(r['confidence'])
                entry['losses'].append(r['loss'])
                entry['corrects'].append(r['correct'])
                entry['softmax_alls'].append(r['softmax_all'])
                entry['sources'].append({'round': r['round'],
                                         'fold': r['fold']})

    print(f"\n[汇总] 加载了 {loaded_count}/{num_rounds * num_folds} 个折的结果")

    # ── 计算聚合指标 ──
    aggregated = {}
    incomplete_paths = []  # num_predictions != num_rounds 的路径
    for path, entry in image_results.items():
        n = len(entry['pred_probs'])
        if n == 0:
            print(f"  [错误] 图片无预测结果: {path}")
            continue

        pred_correct_rate = sum(entry['corrects']) / n
        mean_confidence = np.mean(entry['pred_probs'])

        # 计算平均熵
        entropies = []
        for softmax_vec in entry['softmax_alls']:
            p = np.array(softmax_vec)
            p = np.clip(p, 1e-10, 1.0)
            entropy = -np.sum(p * np.log(p))
            entropies.append(entropy)
        mean_entropy = np.mean(entropies) if entropies else 0.0

        avg_loss = np.mean(entry['losses'])
        std_confidence = np.std(entry['pred_probs']) if n > 1 else 0.0

        aggregated[path] = {
            'image_path': path,
            'sample_id': entry['sample_id'],
            'patient_id': entry['patient_id'],
            'true_label': entry['true_label'],
            'pred_correct_rate': round(pred_correct_rate, 6),
            'mean_confidence': round(mean_confidence, 6),
            'mean_entropy': round(mean_entropy, 6),
            'avg_loss': round(avg_loss, 6),
            'num_predictions': n,
            'std_confidence': round(std_confidence, 6),
            'all_confidences': json.dumps(
                [round(x, 6) for x in entry['pred_probs']]),
            'all_correct_flags': json.dumps(entry['corrects']),
            'all_losses': json.dumps(
                [round(x, 6) for x in entry['losses']]),
            'prediction_sources': json.dumps(entry['sources']),
        }

        # 强校验: 每张图应恰好获得 num_rounds 次预测
        if n != num_rounds:
            incomplete_paths.append((path, n))

    # ── 报告预测次数不完整的样本 ──
    if incomplete_paths:
        # 按缺失程度排序，缺失最多的排前面
        incomplete_paths.sort(key=lambda x: x[1])
        print(f"\n  [警告] {len(incomplete_paths)} 张图片的预测次数 "
              f"!= {num_rounds} (应为每张图 {num_rounds} 次):")
        for p, n in incomplete_paths[:20]:
            print(f"    {n}/{num_rounds} 次: {p}")
        if len(incomplete_paths) > 20:
            print(f"    ... 共 {len(incomplete_paths)} 张")

    # ── 检查是否有图片完全缺少预测 ──
    all_path_set = set(all_image_paths)
    covered_path_set = set(aggregated.keys())
    missing = all_path_set - covered_path_set
    if missing:
        print(f"\n  [错误] {len(missing)} 张图片完全缺少 OOF 预测结果!")
        for p in sorted(missing)[:10]:
            print(f"    - {p}")
        if len(missing) > 10:
            print(f"    ... 共 {len(missing)} 张")

    return aggregated


def assign_grade(pred_correct_rate, mean_confidence, args):
    """
    五级分级: A / B / C / D / E

    判级顺序（先判高质量再判低质量，C 为默认）:
      A → B → E → D → C

    A: 核心高纯样本 (correct_rate >= 0.95 AND confidence >= 0.95)
    B: 高质量样本 (correct_rate >= 0.85 AND confidence >= 0.85)
    E: 强疑似噪声 (correct_rate < 0.30 OR confidence < 0.30)
    D: 低质量/高不确定 (correct_rate < 0.60 OR confidence < 0.60)
    C: 中间/边界样本 (其余)
    """
    if (pred_correct_rate >= args.grade_a_acc and
            mean_confidence >= args.grade_a_conf):
        return 'A'
    elif (pred_correct_rate >= args.grade_b_acc and
          mean_confidence >= args.grade_b_conf):
        return 'B'
    elif (pred_correct_rate < args.grade_e_acc or
          mean_confidence < args.grade_e_conf):
        return 'E'
    elif (pred_correct_rate < args.grade_d_acc or
          mean_confidence < args.grade_d_conf):
        return 'D'
    else:
        return 'C'


# ═══════════════════════════════════════════════════════════════════════
# 9. 输出与统计
# ═══════════════════════════════════════════════════════════════════════

def save_final_csv(aggregated, args):
    """保存样本级分级 CSV"""
    fieldnames = [
        'image_path', 'patient_id', 'sample_id', 'true_label',
        'pred_correct_rate', 'mean_confidence', 'mean_entropy',
        'avg_loss', 'num_predictions', 'std_confidence',
        'grade',
        'all_confidences', 'all_correct_flags', 'all_losses',
        'prediction_sources',
    ]

    rows = []
    for path in sorted(aggregated.keys()):
        entry = aggregated[path]
        entry['grade'] = assign_grade(
            entry['pred_correct_rate'],
            entry['mean_confidence'],
            args)
        rows.append(entry)

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[输出] 样本分级 CSV: {args.output_csv} ({len(rows)} 行)")
    return rows


def save_patient_summary_csv(aggregated, args):
    """保存患者级汇总 CSV"""
    patient_data = defaultdict(lambda: {
        'patient_id': None,
        'num_images': 0,
        'grades': [],
        'correct_rates': [],
        'confidences': [],
    })

    for path, entry in aggregated.items():
        pid = entry['patient_id']
        grade = assign_grade(
            entry['pred_correct_rate'],
            entry['mean_confidence'],
            args)
        pd = patient_data[pid]
        pd['patient_id'] = pid
        pd['num_images'] += 1
        pd['grades'].append(grade)
        pd['correct_rates'].append(entry['pred_correct_rate'])
        pd['confidences'].append(entry['mean_confidence'])

    fieldnames = [
        'patient_id', 'num_images',
        'num_A', 'num_B', 'num_C', 'num_D', 'num_E',
        'mean_correct_rate', 'mean_confidence',
    ]

    rows = []
    for pid in sorted(patient_data.keys()):
        pd = patient_data[pid]
        gc = Counter(pd['grades'])
        rows.append({
            'patient_id': pid,
            'num_images': pd['num_images'],
            'num_A': gc.get('A', 0),
            'num_B': gc.get('B', 0),
            'num_C': gc.get('C', 0),
            'num_D': gc.get('D', 0),
            'num_E': gc.get('E', 0),
            'mean_correct_rate': round(np.mean(pd['correct_rates']), 4),
            'mean_confidence': round(np.mean(pd['confidences']), 4),
        })

    with open(args.patient_summary_csv, 'w', newline='',
              encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[输出] 患者汇总 CSV: {args.patient_summary_csv} ({len(rows)} 行)")


def print_grading_summary(rows, aggregated, class_indices, args):
    """打印分级统计信息"""
    print("\n" + "=" * 70)
    print("  样本质量分级统计报告")
    print("=" * 70)

    # ── 总体分布 ──
    total = len(rows)
    grade_counts = Counter(r['grade'] for r in rows)
    print(f"\n总样本数: {total}")
    print(f"\n{'Grade':<8} {'数量':>8} {'占比':>10} {'含义':<25}")
    print("-" * 55)
    grade_meanings = {
        'A': '核心高纯样本',
        'B': '高质量样本',
        'C': '中间/边界样本',
        'D': '低质量/高不确定',
        'E': '强疑似噪声',
    }
    for g in ['A', 'B', 'C', 'D', 'E']:
        cnt = grade_counts.get(g, 0)
        pct = cnt / total * 100 if total > 0 else 0
        print(f"  {g:<6} {cnt:>8} {pct:>9.1f}%  {grade_meanings[g]}")

    # ── 按类别分层统计 ──
    label_names = {v: k for k, v in class_indices.items()}
    print(f"\n按类别分层:")
    for label_val in sorted(label_names.keys()):
        label_name = label_names[label_val]
        subset = [r for r in rows if r['true_label'] == label_val]
        sub_total = len(subset)
        if sub_total == 0:
            continue
        sub_grades = Counter(r['grade'] for r in subset)
        print(f"\n  [{label_name}] 共 {sub_total} 张:")
        for g in ['A', 'B', 'C', 'D', 'E']:
            cnt = sub_grades.get(g, 0)
            pct = cnt / sub_total * 100
            print(f"    Grade {g}: {cnt:>6} ({pct:>5.1f}%)")

    # ── 按患者概览 ──
    patient_grades = defaultdict(lambda: defaultdict(int))
    for r in rows:
        patient_grades[r['patient_id']][r['grade']] += 1

    patients_with_E = [pid for pid, gc in patient_grades.items()
                       if gc.get('E', 0) > 0]
    patients_with_D_or_E = [pid for pid, gc in patient_grades.items()
                            if gc.get('D', 0) > 0 or gc.get('E', 0) > 0]

    print(f"\n患者概览:")
    print(f"  总患者数: {len(patient_grades)}")
    print(f"  含 Grade E 的患者: {len(patients_with_E)}")
    print(f"  含 Grade D/E 的患者: {len(patients_with_D_or_E)}")

    # ── 后续使用建议 ──
    print(f"\n{'=' * 70}")
    print("  后续使用建议")
    print("=" * 70)
    n_a = grade_counts.get('A', 0)
    n_ab = n_a + grade_counts.get('B', 0)
    n_abc = n_ab + grade_counts.get('C', 0)
    print(f"""
  1. 核心高纯训练 (Grade A):
     - 样本数: {n_a} ({n_a/total*100:.1f}%)
     - 特征清晰、典型性强，可获得最稳定的收敛

  2. 高质量训练主集 (Grade A + B):
     - 样本数: {n_ab} ({n_ab/total*100:.1f}%)
     - 兼顾质量和数量，推荐作为主训练集

  3. 兼顾泛化的推荐训练集 (Grade A + B + C):
     - 样本数: {n_abc} ({n_abc/total*100:.1f}%)
     - C 类边界样本对模型泛化能力至关重要

  4. 低质量样本 (Grade D):
     - 样本数: {grade_counts.get('D', 0)}
     - 建议降权训练或用于难例分析

  5. 强疑似噪声 (Grade E):
     - 样本数: {grade_counts.get('E', 0)}
     - 优先人工复核，不建议直接作为主训练样本
""")


# ═══════════════════════════════════════════════════════════════════════
# 10. 主函数
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.base_seed)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ── Step 1: 数据读取与患者映射 ──
    print("\n" + "=" * 50)
    print("Step 1: 数据读取与患者 ID 映射")
    print("=" * 50)

    sample_to_patient = load_csv_mapping(args.csv_path)
    print(f"  CSV 映射: {len(sample_to_patient)} 条 sample→patient 记录")

    samples, class_indices = scan_train_samples(
        args.data_path_train, sample_to_patient)
    all_image_paths = [s['image_path'] for s in samples]

    patient_groups, patient_majority = build_patient_groups(samples)
    num_patients = len(patient_groups)
    print(f"  患者: {num_patients} 名")
    print(f"  每患者图像数: "
          f"min={min(len(v) for v in patient_groups.values())}, "
          f"max={max(len(v) for v in patient_groups.values())}, "
          f"median={int(np.median([len(v) for v in patient_groups.values()]))}")

    # ── Step 2: 多轮 K-Fold OOF 评估 ──
    model_name = f"{args.model_module.split('.')[-1]}_{args.model_function}"

    print("\n" + "=" * 50)
    print(f"Step 2: 多轮 K-Fold OOF 评估 "
          f"({args.num_rounds} 轮 × {args.num_folds} 折 = "
          f"{args.num_rounds * args.num_folds} 个模型)")
    print("=" * 50)

    total_start = time.time()
    config_dict = {
        'model_module': args.model_module,
        'model_function': args.model_function,
        'num_classes': args.num_classes,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'img_size': args.img_size,
        'num_folds': args.num_folds,
        'num_rounds': args.num_rounds,
        'base_seed': args.base_seed,
    }

    for round_idx in range(args.num_rounds):
        seed = args.base_seed + round_idx
        set_seed(seed)

        folds = build_patient_folds(
            patient_majority, args.num_folds, seed)

        print(f"\n── Round {round_idx} (seed={seed}) ──")

        for fold_idx, (train_pids, val_pids) in enumerate(folds):
            fold_dir = get_fold_dir(
                args.output_dir, model_name, round_idx, fold_idx)

            # 构建训练集和验证集的路径与标签
            train_paths, train_labels = flatten_patient_subset(
                train_pids, patient_groups)
            val_paths, val_labels = flatten_patient_subset(
                val_pids, patient_groups)
            val_samples = [
                s for s in samples
                if s['patient_id'] in set(val_pids)
            ]

            # 断点续训检查
            expected_count = len(val_paths)
            if check_fold_completed(fold_dir, expected_count):
                print(f"  [跳过] R{round_idx}/F{fold_idx}: "
                      f"已完成 ({expected_count} 条)")
                continue

            print(f"\n  Round {round_idx}, Fold {fold_idx}:")
            print(f"    训练: {len(train_pids)} 患者, "
                  f"{len(train_paths)} 张图")
            print(f"    验证: {len(val_pids)} 患者, "
                  f"{len(val_paths)} 张图")

            fold_results = train_one_fold(
                train_paths, train_labels,
                val_paths, val_labels,
                val_samples, args,
                round_idx, fold_idx,
                num_train_patients=len(train_pids),
                train_pids=train_pids,
                patient_groups=patient_groups,
                patient_majority=patient_majority)

            # 保存结果
            save_fold_results(fold_dir, fold_results, config_dict)

    total_time = time.time() - total_start
    print(f"\n[完成] 所有折训练+预测完成，耗时 "
          f"{str(timedelta(seconds=int(total_time)))}")

    # ── Step 3: 汇总预测结果 ──
    print("\n" + "=" * 50)
    print("Step 3: 汇总 OOF 预测结果并分级")
    print("=" * 50)

    aggregated = aggregate_predictions(
        args.output_dir, model_name,
        args.num_rounds, args.num_folds,
        all_image_paths)

    if len(aggregated) != len(all_image_paths):
        print(f"  [警告] 聚合结果 ({len(aggregated)}) 与 "
              f"总样本数 ({len(all_image_paths)}) 不一致!")

    # ── Step 4: 分级与输出 ──
    print("\n" + "=" * 50)
    print("Step 4: 样本分级与输出")
    print("=" * 50)

    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    rows = save_final_csv(aggregated, args)
    save_patient_summary_csv(aggregated, args)
    print_grading_summary(rows, aggregated, class_indices, args)


if __name__ == '__main__':
    main()
