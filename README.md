# Classification — CLE 胃部图像三分类（清洁版）

本仓库于 2026-10-23 从 `../classifyML` **复制**提取而来（原项目保持不动，作为历史归档）。
只包含正式训练/评估链路的最小依赖集合，不带历史权重、输出产物与探索性脚本。

## 任务

共聚焦激光显微内镜（CLE）胃部图像三分类：
0 = Benign_lesions（良性），1 = Malignant_lesions（恶性），2 = body_of_stomach_normal（正常）
映射见 `class_indices.json`。数据划分：train 4977 / val 538 / test 671（按患者划分防泄漏）。

## 环境

conda 环境：`sunfuxiang`（python: `/home/login/miniconda3/envs/sunfuxiang/bin/python`）
依赖见 `requirements.txt`（相对原版补充了 scikit-image）。

## 文件说明

| 文件 | 作用 |
|---|---|
| `train_cle_fair.py` | **训练入口**（fair 公平训练协议：AdamW + Cosine + AMP，patience 早停） |
| `evaluate.py` | **三分类评估入口**（test 集指标 + 混淆矩阵） |
| `cle_fair_common.py` | 共享底座：模型注册表 MODEL_REGISTRY、模型构建与验证变换 |
| `consensus_manifest.py` | 三模型共识样本过滤（`--manifest` 时启用） |
| `train.py` | 仅作库使用：提供 `get_or_compute_mean_std` 等函数（勿直接运行） |
| `utils.py` / `my_dataset.py` | 数据读取与 Dataset |
| `models/` | 模型库（含 `optimize/` 自研改进：resnet34_degconv_cpca_v3 = ResNet34+CPCA） |
| `attention/` | 注意力模块（部分模型引用） |

## 运行示例
> 提示：项目根目录的 `run.sh` 已固定使用 sunfuxiang 环境解释器（本机 shell 的 ~/bin/python 软链指向 dinov2 环境，直接敲 python 会缺依赖）：> `./run.sh train ...` / `./run.sh eval ...` / `./run.sh python xxx.py`

数据集已就位（见下方"说明"），ImageNet 预训练权重仍引用原位置：

```bash
# 训练（在仓库根目录）
python train_cle_fair.py --model resnet34 --num_classes 3 --img_size 224 --seed 42 \
  --data-path-train ./datasets/cle/train \
  --data-path-val   ./datasets/cle/val \
  --weights /home/login/data/cle/sunfuxiang/classifyML/pth/resnet34-333f7ec4.pth \
  --save-dir ./weights --epochs 300

# 评估
python evaluate.py --model resnet34 \
  --weights weights/resnet34_3class_seed42_best.pth \
  --data-path ./datasets/cle/test
```

自研模型（Ours）：`--model resnet34_degconv_cpca_v3 --img_size 512`。
变体模型注册名带版本号（如 `resnet34_s2d_v1` / `resnet34_s2d_v2`、`resnet34_se`），训练权重自动存到 `weights/<模型名>/`，同方法不同版本不会互相覆盖。
注意：注册名沿用历史名，v3 实现已移除 DEGConv、仅保留 CPCA（名实差异沿袭原项目）。

## 说明
- 本仓库 `datasets/cle/` 数据构成（2026-10-23 整理）：
  - **val**（538 张）与 **test**（671 张）：从 `../datasets/3class` 复制，逐类核对一致；
  - **train**（2386 张）：由 demo 五类归并移动而来——萎缩(779)+肠化(150)→Benign_lesions(929)，异型增生(327)+癌变(237)→Malignant_lesions(564)，正常(893)→body_of_stomach_normal(893)。其中 2318 张经文件名核对为原 `3class/train`(4977) 子集且类别归属逐张一致，另 68 张（萎缩44+正常24）为原数据集未有的新增图片；类别归并关系与原项目三分类定义一致。

- 首次训练会在仓库根目录生成 `norm_cache/`（Mean/Std 缓存，已被 .gitignore 排除）。
- 原 `train.py` 内的旧路径默认值指向已迁移的 `/home/login/cle/...`；作为库导入不受影响，如需直接运行请用参数覆盖。
- 历史实验（Full vs HQ、共识消融、CUFIT 对照等）的完整证据链在 `../classifyML` 与 `../PROJECT_AUDIT.md`。
