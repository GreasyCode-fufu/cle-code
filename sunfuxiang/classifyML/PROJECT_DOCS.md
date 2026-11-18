# classifyML 项目全文件文档

> 本文档覆盖 `classifyML` 项目中的全部代码文件与重要数据/产物文件，按目录组织。
> 日常操作命令请配合已有的 [GUIDE.md](GUIDE.md)（使用指南）阅读，本文档侧重"每个文件是什么、干什么、与谁关联"。
> 文档生成时间：2026-09-28。

---

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构总览](#2-目录结构总览)
3. [根目录核心脚本](#3-根目录核心脚本)
4. [models/ 模型库](#4-models-模型库)
5. [attention/ 注意力模块库](#5-attention-注意力模块库)
6. [bow_fusion/ 双流融合框架](#6-bow_fusion-双流融合框架)
7. [heatMap/ 热力图工具](#7-heatmap-热力图工具)
8. [visualization/ 绘图脚本](#8-visualization-绘图脚本)
9. [mobileFormer/ 模型包](#9-mobileformer-模型包)
10. [scripts/ 数据与实验辅助脚本](#10-scripts-数据与实验辅助脚本)
11. [tests/ 单元测试](#11-tests-单元测试)
12. [共识消融实验（manifests / audit / results）](#12-共识消融实验manifests--audit--results)
13. [Shell 脚本](#13-shell-脚本)
14. [数据、结果与输出目录](#14-数据结果与输出目录)
15. [核心数据流](#15-核心数据流)
16. [已知问题与注意事项](#16-已知问题与注意事项)

---

## 1. 项目概述

### 1.1 研究背景与任务

本项目是一个基于 PyTorch 的**共聚焦激光显微内镜（CLE）胃部图像分类**研究仓库。核心任务为三分类：

| 类别索引 | 类别名 | 含义 |
|---:|---|---|
| 0 | `Benign_lesions` | 良性病变 |
| 1 | `Malignant_lesions` | 恶性病变 |
| 2 | `body_of_stomach_normal` | 正常胃体 |

固定数据划分：train = 4977 张，val = 538 张，test = 671 张（test 三类为 256 / 210 / 205）。类别索引映射由脚本自动写入根目录 `class_indices.json`。

历史上还做过二分类（abnormal/normal）与五分类的探索性实验，对应 `outputs_2class_*`、`outputs_5class_*` 等目录。

### 1.2 三条主要实验主线

1. **数据质量对照实验（Full vs HQ）**：全量 4977 张 vs "三模型共识"筛出的高质量 2880 张（`datasets/3class_highquality`），6 个基线模型 + 1 个自研模型（Ours = ResNet34 + DEGConv + CPCA），seeds 42/43/44。结论见 `analysis_3seed_summary.md`：HQ 全面优于 Full，Ours 达 91.05±1.00 Macro-F1，超过全部基线。
2. **共识消融实验（consensus ablation，2026-09）**：追问"HQ 的收益到底来自哪个筛选器"。固定 ResNet34 分类器与全部超参，仅改变训练集合（Full / R / V / C / RV / RC / VC / RVC 八组 × 3 seeds）。结论：最优为 **RC（0.9124）而非三模型 RVC（0.9011）**，所有筛选组均优于 Full（0.8764），Malignant 类 recall 受益最大。详见第 12 节。
3. **双流融合实验（bow_fusion/）**：LBP-BoW 纹理特征 + CNN 深度特征的 Cross-Attention 双流网络，复用 fair 公平训练协议，与单流模型同口径评估对比。

### 1.3 技术栈与环境

- conda 环境 `pytorch-2.1.1`（Python 3.11，CUDA）；依赖见 `requirements.txt`（torch 2.1.1 / torchvision 0.16.1、timm、sklearn、xgboost、lightgbm、thop、torchinfo、rich、joblib 等；注意 `scikit-image` 实际被 `lbp_ml_classify.py` 和 `bow_fusion/` 依赖但未列入清单）。
- 训练协议（fair 框架统一）：AdamW（lr 1e-4，wd 0.05）+ CosineAnnealingLR + AMP，batch 32，epochs 300（early stop 30，patience 按 val macro-F1），ImageNet 归一化（有预训练权重时），224 输入（Ours 模型用 512）。

---

## 2. 目录结构总览

```
classifyML/
├── train.py / KFold.py / evaluate*.py / train_cle_fair.py ...   # 核心训练评估脚本（第 3 节）
├── cle_fair_common.py / consensus_manifest.py / utils.py / my_dataset.py  # 共享库
├── models/        # 约 61 个模型文件 + optimize/ 改进主线（第 4 节）
├── attention/     # 25 个注意力模块（第 5 节）
├── bow_fusion/    # BoW+CNN 双流融合框架（第 6 节）
├── heatMap/       # Grad-CAM 可视化（第 7 节）
├── visualization/ # 论文绘图脚本（第 8 节）
├── mobileFormer/  # Mobile-Former 独立模型包（第 9 节）
├── scripts/       # 12 个辅助脚本（第 10 节）
├── tests/         # 共识消融单元测试（第 11 节）
├── manifests/     # 共识消融训练清单（第 12 节）
├── consensus_ablation_audit/    # 消融实验预注册/审计文档（第 12 节）
├── consensus_ablation_results/  # 消融实验最终结果（第 12 节）
├── archive/       # 历史遗留脚本（第 3.8 节）
├── outputs*/      # KFold 分级与评估输出（第 14 节）
├── weights*/      # 各实验权重产物（第 14 节）
├── pth/           # ImageNet 预训练权重（第 14 节）
├── norm_cache/    # Mean/Std 缓存（第 14 节）
├── GUIDE.md       # 使用指南（命令示例）
└── PROJECT_DOCS.md  # 本文档
```

---

## 3. 根目录核心脚本

### 3.1 公平对比框架（"fair 三件套"，当前正式训练评估链路）

#### `cle_fair_common.py`（211 行）—— 共享底座

fair 框架的**单一事实来源**公共库，被训练端与评估端共同依赖，保证模型名与评估协议对齐；刻意不修改任何老框架文件（`train.py` / `utils.py` / `evaluate_2class.py`）。

- `MODEL_REGISTRY`：约 40 个模型名 → (模块, 工厂函数) 注册表。覆盖 FasterNet、ResNet、DenseNet、ViT 全家、Swin（含 vanilla 对照键）、PVT、RepViT，以及 `models/optimize` 下的 `resnet34_sa`、`degconv_sa`、`degconv_cpca_v1~v3`（即 Ours 系列）。
- `build_model(name, num_classes)`：动态导入并实例化模型。
- `read_images_dir(root)`：扫描 `root/<class>/*`，类序 = 子目录名排序（negative=0, positive=1）。
- `build_val_transform(img_size)`：强制方形 Resize + CenterCrop + Normalize（CUFIT 协议，训练选模与测试评估共用，避免预处理漂移）。
- `run_inference()`：AMP autocast + softmax 概率矩阵。
- `youden_threshold()`：正类概率上以 0.01 步长扫 0.05~0.95，最大化 J = sens + spec − 1。
- `compute_binary_metrics()` / `compute_per_class_metrics()`：完整二分类指标字典（acc / macro 与 weighted P/R/F1 / AUROC / sens / spec / AUC-PR / kappa / MCC / paper_f1 / PPV / NPPV / 混淆矩阵等）。

被 `train_cle_fair.py`、`evaluate_cle_fair.py`、`extract_and_classify_mult.py`、`verify_weights_loading.py`、`bow_fusion/` 引用。

#### `train_cle_fair.py`（523 行）—— 训练端（**所有正式训练的唯一入口**）

与 CUFIT 评估协议对齐的公平训练脚本，在 `train.py` 基础上做四点改动：(1) `--model` 走注册表而非硬编码 import；(2) 最优 checkpoint / 早停按 **val macro-F1**（而非 val_acc）；(3) 保存 `run_meta.json`（含归一化 mean/std、模型、img_size 等）供评估端复用；(4) tqdm.rich 金色进度条。

关键函数：
- `load_pretrained_weights_strict()`：严格预训练加载——按顶层模块名 head/fc/classifier 删除分类头键；对 vanilla Swin 仅允许忽略固定的 5 个 attn_mask buffer；missing/unexpected 与期望严格匹配否则 `RuntimeError`；返回审计信息（checkpoint sha256、被删键等）。
- `prepare_training_samples()`：先按原逻辑扫描 Full 训练集，再可选应用 manifest 过滤（调 `consensus_manifest.filter_training_samples`，并把 val/test 作为排除根做泄漏检查）。这是共识消融实验接入点（`--train-manifest` 参数）。
- `build_run_meta()`：写 run_meta.json；manifest 模式下写入约 27 个可复现字段（manifest_sha256 / optimizer / 早停 / 预训练哈希 / 代码哈希等），无 manifest 时保持旧 13 字段向后兼容。
- `evaluate_val_macro_f1()`：验证集返回 (loss, acc, macro_f1)。

主要参数：`--model`(fast_t0)、`--num_classes`(2)、`--epochs`(300)、`--batch-size`(32)、`--lr`(1e-4)、`--label-smoothing`(0)、`--img_size`(224)、`--data-path-train`、`--train-manifest`('')、`--data-path-val`、`--weights`、`--device`、`--save-dir`、`--save-name`(model_best.pth)、`--csv-name`(train_log.csv)、`--early-stop`(30)。

输出：`<save-dir>/<model>[-N]/{model_best.pth, train_log.csv（含 val_macro_f1 列）, run_meta.json}`。

#### `evaluate_cle_fair.py`（259 行）—— 评估端（二分类）

CUFIT 协议的二分类公平评估。流程：归一化解析（CLI > 权重旁 run_meta.json，找不到直接抛错不兜底）→ build_model + 严格加载 best 权重 → 在 **VAL** 上 Youden's J 扫阈值 → 在 **TEST** 上输出两套指标（阈值优化 vs 默认 0.5）→ 盒子格式报告 + JSON。

参数：`--model`、`--weights`、`--data-path-val`、`--data-path-test`、`--img_size`、`--batch-size`、`--class-names`、`--norm-mean/--norm-std`、`--run-meta`、`--out-dir`、`--device`。

输出：`<out-dir 或权重目录>/eval_cle_fair_results.json`（含 best_threshold、youden_j、两套指标）。

### 3.2 早期训练与评估脚本（老框架，仍保留可用）

#### `train.py`（455 行）

Baseline 单模型训练脚本（文件内硬编码 import `models.FasterNet.fast_t0`，切换模型需改 import 行）。特点：
- `set_seed()` 固定随机源；
- `compute_dataset_mean_std()` / `get_or_compute_mean_std()`：从**训练集**自动计算 Mean/Std，缓存键为 `md5(数据路径)_img{尺寸}`，并用数据 fingerprint（所有图片相对路径+大小+mtime 的 SHA256）做失效校验（`--recompute-norm` 强制重算，`--norm-cache-dir` 指定缓存目录）；
- thop 统计 MACs/FLOPs/Params + torchinfo 打印架构；
- AdamW + CosineAnnealingLR + AMP；**按 val_acc 保存最优**（与 fair 框架的 macro-F1 选优不同）；支持 `--freeze-layers` 与早停。

依赖：`my_dataset.MyDataSet`、`utils.read_split_train_data/read_split_val_data/train_one_epoch/evaluate`。输出：`<save-dir>/<模型名>[-N]/<save-name>.pth` + 逐 epoch CSV 日志 + norm 缓存。

#### `evaluate.py`（526 行）—— 多分类评估（"规范版 v2"，**三分类主线评估脚本**）

主指标 Macro-F1 的全面测试集评估。
- `MODEL_REGISTRY`：模型名 → (模块, 工厂函数)，含 resnet / repvit / fast_t0 / pvt / vanillanet / alexnet / convnext / efficientnet / mobilenet / swin / vit 及 `models.optimize` 下的 resnet34_sa / degconv / cpca 变体。
- `resolve_norm()`：归一化三级解析——CLI > 权重旁 `run_meta.json` > ImageNet 兜底（警告）。
- `compute_complexity()`：thop 复杂度统计；`run_inference()`：区分纯推理与端到端 FPS。
- `compute_metrics()`：混淆矩阵、acc、balanced acc、kappa、MCC、macro/weighted P-R-F1、每类 OvR Sens/Spec/Prec/F1/NPV、每类 AUROC 与 AUC-PR。
- `bootstrap_ci()`：1000 次重采样 95% CI（`--bootstrap` 开启）。

参数：`--model`（必填，choices=注册表）、`--weights`、`--data-path`、`--class-names`（默认 Benign_lesions Malignant_lesions body_of_stomach_normal）、`--img-size`(224)、`--norm-mean/--norm-std`、`--batch-size`(64)、`--out-dir`、`--bootstrap`、`--n-bootstrap`(1000)。

输出：`<out-dir 或权重目录>/eval_<时间戳>/{predictions.csv, results.json, evaluation_report.md}`。`run_eval_all.py` 与共识消融汇总脚本都依赖 `results.json` / `predictions.csv` 的字段。

#### `evaluate_2class.py`（718 行）—— 二分类评估（老版）

除指标外还绘制混淆矩阵 / ROC / PR / 每类指标四张 PNG。模型注册表较旧（18 个）。正类 = index 1；`paper_f1 = 2·WP·WR/(WP+WR)`。输出 `./outputs/eval_binary_<model>_<时间戳>/`。

### 3.3 `KFold.py`（1393 行）—— 训练集样本质量分级系统

文件内部名 `dataset_quality_grading.py`。**基于"患者级 K 折 OOF 预测 + 多轮种子重复"给每张训练图像打 A/B/C/D/E 五级质量标签**，是共识消融实验 R/V 筛选器的数据来源。

核心流程（main 四步）：
1. **数据与患者识别**：读 `--csv-path`（train_processed.csv，split=='train' 行）建立 sample_id → 患者ID 映射；`scan_train_samples()` 递归扫描训练目录（兼容扁平/按患者嵌套）。
2. **K 折训练**：`build_patient_folds()` 用 sklearn **StratifiedGroupKFold**（患者为 group、患者多数类为分层标签，保证同患者同折防泄漏）；`num_rounds` × `num_folds` 个模型（每轮 seed = base_seed + round）；`train_one_fold()` 支持内层患者级早停（monitor 默认 macro_f1）、按折内标签计算 balanced 类别权重、动态导入 `--model-module/--model-function` 构建（默认 models.Resnet.resnet34）、ImageNet 归一化（匹配预训练）；每折输出逐样本 OOF 预测（pred_label / pred_prob / confidence / loss / correct / softmax_all）。
3. **聚合**：`aggregate_predictions()` 按图片聚合所有轮×折预测 → correct_rate、mean_confidence、mean_entropy、avg_loss 等（强校验每图恰好 num_rounds 次预测）。
4. **分级**：`assign_grade()` 判级顺序 A→B→E→D→C：
   - **A**：correct_rate ≥ 0.95 且 confidence ≥ 0.95
   - **B**：correct_rate ≥ 0.85 且 confidence ≥ 0.85
   - **E**：correct_rate < 0.30 或 confidence < 0.30（强疑似噪声）
   - **D**：correct_rate < 0.60 或 confidence < 0.60
   - **C**：其余（边界样本）
   阈值均可由 `--grade_{a,b,d,e}_{acc,conf}` 覆盖。

输出：每折 `oof_predictions.json`（断点续训粒度，存在且完整即跳过重训）、`train_sample_grading.csv`（样本级分级）、`train_patient_grading_summary.csv`（患者级汇总）。**不保存模型权重**（目的是评估样本质量而非产出部署模型）。

> 备份文件 `KFold.py.bak` ~ `.bak4` 为历史迭代版本，仅存档用。

### 3.4 共享工具库

#### `utils.py`（328 行）

老框架公共工具（fair 框架刻意不修改它）：
- `read_split_train_data()` / `read_split_test_data()` / `read_split_val_data(root, val_rate=1.0)`：扫描 `root/<类名>/`（类名排序编号），返回 (路径列表, 标签列表)；train/test 版本会把 类别索引→类名 写入 `class_indices.json`（副作用）。
- `read_split_data(root, val_rate=0.2)`：单目录随机划分 train/val。
- `plot_data_loader_image()`：可视化一个 batch（反 ImageNet 归一化）。
- `write_pickle()` / `read_pickle()`。
- `train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, class_weights=None)`：单 epoch 训练——可选类别加权 CE、AMP、tqdm 进度条、非有限 loss 时退出；返回 (avg_loss, acc)。**KFold.py 用的就是它的 class_weights 版本**。
- `evaluate()` / `test()`：`@torch.no_grad()` 验证/测试循环（返回 avg_loss, acc）。

> `utils.py.bak` / `.bak2` 为历史备份。

#### `my_dataset.py`（104 行）

- `MyDataSet(Dataset)`：通用图像分类数据集——传入路径列表 / 标签列表 / transform，`__getitem__` 打开图片转 RGB 并应用变换；静态 `collate_fn`。被 train.py / KFold.py / train_cle_fair.py / extract_and_classify*.py / bow_fusion 广泛使用。
- `DriveDataset(Dataset)`：DRIVE 视网膜血管分割数据集（历史遗留，与分类主线无关）。

#### `consensus_manifest.py`（203 行）—— 共识消融的校验与过滤核心

manifest（训练清单 CSV）的校验与过滤工具库：
- `MANIFEST_FIELDS = ("sample_id", "class_name", "relative_path")`；`SAMPLE_ID_RE`（`S_十六进制`）；`sha256_file()`（分块 SHA256，用于 run_meta 审计）。
- `scan_sample_ids(roots)`：扫描排除根（val/test）收集 sample_id，跨根重复即报错。
- `filter_training_samples()`：核心——对 Full 扫描结果建立 relative_path 与 sample_id 双索引 → 逐行校验 manifest（表头精确匹配、查重、class_name 与路径首段一致、路径存在且在 Full 内、sample_id 与路径双向匹配）→ 与排除根求交集（有交集即泄漏报错）→ 按原扫描顺序过滤 → 数量必须等于 manifest 行数 → 返回 (paths, labels, info)。
- 拒绝绝对路径与 `..` 穿越（`_safe_relative_path` / `_resolved_inside`）。

被 `train_cle_fair.py` 与 `scripts/` 下三个消融脚本依赖。

### 3.5 特征提取 + 传统机器学习分类支线

#### `extract_and_classify.py`（352 行）

早期二分类端到端测试脚本（硬编码 resnet34 与归一化 0.2939/0.2064，类别 abnormal/normal）。加载权重 → 测试集推理计时 → 指标 → `predictions.csv` + `evaluation_report.md`。依赖 `models.Resnet.resnet34`、`my_dataset`、`utils`。

#### `extract_and_classify_mult.py`（778 行）

多分类（任意 N 类）扩展版，已接入 fair 框架：
- `_prepare_feature_extractor()`：按优先级自动移除分类头（VanillaNet cls1+cls2 → deploy cls → 通用最后 Linear→Identity → 兜底 1×1 Conv+Pool+Flatten），返回 (feat_dim, 策略名)。
- 归一化三级解析：CLI > run_meta.json（与权重同目录）> 重算训练集（复用 train.py 的 fingerprint 缓存机制）。
- 流程：模型名校验 → 类别自动检测 → 构建特征提取器 → E2E 推理与计时 → E2E 指标管线 → 保存 `test_features.npy`。
- 输出：`<output-dir>/ml_classify_<时间戳>/<model>_E2E/`（predictions.csv + evaluation_report.md）+ 特征 npy。
- 依赖：`my_dataset`、`utils`、`cle_fair_common`。

#### `lbp_ml_classify.py`（593 行）—— 传统 ML 基线（无深度学习）

三种 LBP 纹理特征配置 × 三种经典分类器的二分类对比实验，作为深度模型对照组：
- `LBP_CONFIGS`：**LBP-10**（P=8,R=1 全图 10 维）、**LBP-54**（三尺度 (8,1)+(16,2)+(24,3) 拼接 54 维）、**LBP-486**（3×3 分块 × 三尺度，类比 ResNet 512 维）。
- 分类器：Logistic Regression（max_iter=1000）/ SVM-Linear（probability=True）/ RandomForest（200 棵，max_depth=8）。
- joblib loky 并行提特征；StandardScaler 标准化。
- 输出：`<output-dir>/lbp_ml_<时间戳>/<LBP-10|54|486>/<方法>/`（predictions.csv, report.md）+ 各配置特征 npy + `all_configs_comparison.csv`。

### 3.6 批量调度与自检

#### `run_eval_all.py`（330 行）

Full vs HQ 实验的批量评估调度器：遍历 **3 seed(42/43/44) × 2 数据集(hq/full) × 6 模型 = 36 组合**，subprocess 调用 `evaluate.py`（归一化自动读 run_meta.json），已有完整结果则跳过，最后汇总三张 CSV。
- 权重命名约定：`weights_3class_{ds}_seed{seed}/<model>/<model>_3class_{ds}_seed{seed}_best.pth`。
- 输出：`summary_hq.csv`、`summary_full.csv`、`summary_compare.csv`（每模型 hq/full 的 mean±std 与 diff，macro_f1 在最前）。
- 参数：`--data-path`、`--out-dir`、`--device`、`--batch-size`、`--summary-only`、`--force`、`--dry-run`。

#### `verify_weights_loading.py`（74 行）

预训练权重加载自检：硬编码校验 `pth/` 下三份权重（fast_t0 / resnet34 / densenet121）能否被 `cle_fair_common.build_model` 构建的模型正确加载（CPU，strict=False，打印 missing/unexpected 数量），任一失败 `sys.exit(1)`。无参数。

### 3.7 根目录数据与结果文件

| 文件 | 说明 |
|---|---|
| `class_indices.json` | 类别索引映射 `{"0":"Benign_lesions","1":"Malignant_lesions","2":"body_of_stomach_normal"}`，由 `utils.read_split_train/test_data` 每次运行刷新 |
| `requirements.txt` | Python 依赖清单（torch 2.1.1 / torchvision 0.16.1 / sklearn / xgboost / lightgbm / thop / torchinfo / tqdm / rich / joblib 等；缺 scikit-image） |
| `GUIDE.md` | 中文使用指南：环境配置、各脚本命令行示例、典型工作流程 |
| `analysis_3seed_summary.md` | **Full vs HQ 对照实验的论文级总结**（新增 CUFIT Full 与表9，共 9 张表及种子附录）：6 基线在 HQ 上全部优于 Full（+1.86~+2.81 pp，RepViT-M1 +7.42 pp）；**Ours（ResNet34+DegConv+CPCA，仅训 HQ）91.05±1.00 Macro-F1 / 98.03 AUROC，超过全部基线**；提升主要来自 Malignant 类；Ours 21.95M 参数、976 FPS |
| `analysis_3seed_overall.csv` / `analysis_3seed_perclass.csv` | 上文的机器可读版（总体 / 每类指标，mean±std） |
| `cufit_comparison_results/` | **CUFIT Full 补齐 seed 43/44，seed 42 复测复用**；3-seed Macro-F1 90.43±0.38 / Acc 90.31±0.39 / AUROC 97.97±0.03；与 HQ+Ours 主比较及 RVC/RC 参考，含详细报告、CSV、审计和原始预测归档 |
| `summary_full.csv` / `summary_hq.csv` / `summary_compare.csv` | `run_eval_all.py` 的原始输出（36 行明细 / 对比表） |
| `run_eval_all.log` | `run_eval_all.py` 运行日志（2026-06-19，逐组评估输出） |
| `.claude/settings.local.json` | Claude Code 本地权限配置（工具元数据，与研究无关） |
| `KFold.py.bak~bak4`、`utils.py.bak/.bak2` | 历史备份，仅存档 |

### 3.8 `archive/` —— 历史遗留脚本（4 个）

| 文件 | 说明 |
|---|---|
| `archive/train_old.py` | 旧版训练脚本（实际生效模型为 MobileViT，因连续两次 import 覆盖）；AdamW + 手动 lr×0.1 衰减；含 FPS 基准函数 |
| `archive/train_googlenet.py` | GoogLeNet 训练脚本（主 loss + 0.3×辅助 loss1 + 0.3×辅助 loss2，coffee 3 类） |
| `archive/predict_old.py` | 单图预测（swin_tiny + flower 5 类，输出各类概率） |
| `confusion_matrix_old.py` | `ConfusionMatrix` 类：累加混淆矩阵 + PrettyTable 指标 + 热图绘制；顶部保留十余个模型切换注释，曾用于多模型横向对比 |

---

## 4. models/ 模型库

### 4.0 调用机制与命名规律

- 无 `models/__init__.py`，按文件名直接引用。训练脚本（KFold.py / train_cle_fair.py / evaluate.py）通过注册表或 `importlib.import_module` + 工厂函数名动态构建，**每个文件的工厂函数名就是引用键**。
- 命名规律：**`*_ET.py` = 在原版模型中嵌入 EFG（Edge Feature Guidance，边缘特征引导）模块的改进版**；`Swin_vanilla.py` = 无 EFG 的受控对照基线；`optimize/` = 最新的 ResNet34 DEGConv+CPCA 改进主线。
- 几乎全部工厂默认 `num_classes=1000`（in21k 版 21843/21841），由训练脚本传入实际类别数覆盖。

### 4.1 经典 / 手写 CNN

| 文件 | 模型 | 工厂函数 / 实例化方式 |
|---|---|---|
| `AlexNet.py` | 经典 AlexNet（内含被注释的 SE/NonLocal 实验痕迹） | 直接 `AlexNet(num_classes)` |
| `VGGNet.py` | VGG11/13/16/19 | `vgg(model_name="vgg16")` |
| `Resnet.py` | ResNet-34/50/101、ResNeXt50_32x4d/101_32x8d | `resnet34 / resnet50 / resnet101 / resnext50_32x4d / resnext101_32x8d (num_classes, include_top)`（**KFold 默认模块**） |
| `DenseNet.py` | DenseNet-121/169/201/161 | `densenet121/169/201/161`，另有 `load_state_dict()` 工具（加载本地权重并剔除分类头） |
| `GoogleNet.py` | GoogLeNet v1（含辅助分类头） | 直接 `GoogleNet(num_classes, aux_logits=True)` |
| `SENet.py` | 手写 SE-ResNet | `SE_Resnet18/50/101/151`（注意 151 为笔误，实为 152 配置） |
| `SqueezeNet.py` | SqueezeNet 1.0/1.1 | `squeezenet1_0/1_1`（权重下载被注释） |
| `MnasNet.py` | MnasNet 0.5~1.3 | `mnasnet0_5/0_75/1_0/1_3`（下载被注释） |
| `MobileNet.py` | MobileNetV2 | 直接 `MobileNetV2(num_classes)` |
| `MobileNetV3.py` | MobileNetV3 | `mobilenet_v3_large / mobilenet_v3_small` |
| `ShuffleNet.py` | ShuffleNetV2 | `shufflenet_v2_x0_5/x1_0/x1_5/x2_0` |
| `GhostNet.py` | GhostNet（GhostModule+SE） | `ghostnet()` |
| `EfficientNet.py` | 手写 EfficientNet-B0~B7 | `efficientnet_b0` ~ `efficientnet_b7` |

### 4.2 现代 / 结构重参数化 CNN

| 文件 | 模型 | 工厂函数 | 备注 |
|---|---|---|---|
| `FasterNet.py` | FasterNet（CVPR2023，PConv） | `fast_t0` | **train.py 默认模型**；timm.layers 依赖 |
| `Convnext.py` | ConvNeXt | `convnext_tiny/small/base/large/xlarge` | 无 timm import |
| `ConvNeXtV2.py` | ConvNeXtV2（GRN） | `convnextv2_atto ~ huge` | 依赖 `models/utils.py`（LayerNorm/GRN） |
| `RepVGG.py` | RepVGG | `create_RepVGG_A0 ~ D2se`，`repvgg_model_convert()` | 重参数化融合 |
| `RepVGGPlus.py` | RepVGG+ | `create_RepVGGplus_L2pse` 等 | |
| `RepViT.py` | RepViT（CVPR2024） | `repvit_m1/m2/m3` | m1 配置已对齐清华 MIG 官方 |
| `ParNet.py` | ParNet | `parnet_sm/md/l/xl (in_channels, num_classes)` | 注意参数顺序 |
| `PoolFormer.py` | PoolFormer | `poolformer_s12 ~ m48` | 少数真正支持自动下载权重者 |
| `MogaNet.py` | MogaNet（含 SSL/MIM 变体） | 无工厂（mmcv cfg 风格） | **硬依赖 mmcv** |
| `MobileOne.py` | Apple MobileOne | `mobileone(variant=...)`，`reparameterize_model()` | |
| `VanilaNet.py` | VanillaNet **+ 每块前插 CBAM（作者改动）** | `vanillanet_5 ~ 13_x1_5_ada_pool` | 依赖 `attention/CBAM.BasicBlock` |

### 4.3 ViT 家族

| 文件 | 模型 | 工厂函数 | 备注 |
|---|---|---|---|
| `Vit.py` | ViT（手写复现） | `vit_base_patch16_224` 等 8 个 | 无 timm 依赖 |
| `Vit_ET.py` | **ViT + EFG**（patch_embed 后过 EFG_Module 再展平） | 同 Vit.py 8 个 | 依赖 `models/block.py` |
| `Deit.py` | DeiT | `deit_tiny/small/base_patch16_224` 及 distilled/384 版 | **直接子类化 timm ViT**；支持 torch.hub 下载 |
| `Cait.py` | CaiT | `cait_XXS24_224` 等 10 个 | timm 依赖 |
| `Coat.py` | CoaT | `coat_tiny ~ coat_lite_medium` | timm 依赖 |
| `Cvt.py` | CvT | `cvt_tiny` | 本地权重加载 |
| `CSWin.py` | CSWin Transformer | `CSWin_64_12211_tiny_224` 等 6 个 | timm 依赖 |
| `T2T.py` | Tokens-to-Token ViT | 无工厂（直接 `T2T(...)`） | |
| `TNT.py` | Transformer in Transformer | `tnt_s/b_patch16_224` | 强 timm 依赖 |
| `Pit.py` | PiT（含蒸馏变体） | `pit_b/s/xs/ti` 及 `_distilled` | 本地 `weights/pit_*.pth` 加载 |
| `crossvit.py` | CrossViT（双分支交叉注意力） | `crossvit_tiny_224` 等 11 个 | 复用 timm Block |
| `shiftVit.py` | ShiftViT | 无工厂 | |
| `transformerSeries.py` | **时间序列 Transformer**（因果卷积，非图像） | 无工厂 | 与主线无关的工具文件 |

### 4.4 Swin 家族（含 EFG 改进主线）

| 文件 | 说明 |
|---|---|
| `Swin.py` | Swin 复现 **+ 硬编码 EFG 分支**（第 2 个 stage 后无条件执行 token↔2D 转换融合）——早期 EFG 实验遗留版本，旧 checkpoint 依赖此结构。工厂 `swin_tiny_patch4_window7_224` 等 8 个 |
| `Swin_vanilla.py` | **干净原版 Swin（受控对照基线，2025-09-28 新建）**。docstring 明确：为受控消融保留无 EFG 基线，旧 Swin+EFG checkpoint 依赖原模块不改动。工厂同上 |
| `Swin_ET.py` | Swin + EFG（patch_embed 后、4 stage 前对 96 维 token 应用，`use_efg` 开关）。依赖 `models/block.py` |
| `swin_efg.py` | Swin + EFG 于**全部 4 stage 之后**、norm 前对最终 7×7×768 特征应用（前 819 行为被注释的旧实现）。依赖 `models/block.py` |
| `Swin_mix.py` | Swin + **ACmix**（CVPR2022 自注意力与卷积集成）：WindowAttention 并联卷积路径，可学习 rate 加权融合 |

### 4.5 金字塔 / 移动端混合架构

| 文件 | 说明 |
|---|---|
| `Pvt.py` | PVT v1；工厂 `pvt_tiny/small/medium/large/huge_v2` |
| `Pvt_ET.py` | **PVT + EFG**（第 4 stage 内 512 维 token 特征瓶颈融合） |
| `Mobilevit.py` | Apple MobileViT 原版；`mobile_vit_xx_small/x_small/small`；结构配置来自 `model_config.py` |
| `Mobilevit_ET.py` | **MobileViT + CBAM**（每个 MobileViTBlock 内嵌 `attention/CBAM.BasicBlock`） |
| `EfficientFormer.py` | `efficientformer_l1/l3/l7` |
| `EfficientFormerV2.py` | `efficientformerv2_s0/s1/s2/l` |

### 4.6 ViG / GNN / 其他

| 文件 | 说明 |
|---|---|
| `Vig.py` | ViG（Vision GNN，Grapher+FFN）；`vig_ti/s/b_224_gelu` |
| `Vig_ET.py` | **ViG + BasicConv 瓶颈重构 + CoordAtt**（分组 1×1 降维 → 3×3 分组卷积 → CoordAtt → 升维）；依赖 `attention/CoordAtt.py` |
| `PVig.py` | 金字塔版 ViG（加 Downsample 四阶段）；`pvig_ti/s/m/b_224_gelu` |
| `PVig_ET.py` | **PVig + FFN 换成带 3×3 深度卷积中间层的 Mlp** |
| `GAT.py` | 图注意力卷积层（torch_geometric 风格教学摘抄，**缺 import，独立运行会 NameError**） |
| `DWNet.py` | **注意力→（可选动态）大核深度卷积替换实验模型**（Swin 骨架 + DWBlock）。三种卷积模式：普通静态深度卷积 / CondConv 式输入相关动态核 / 逐位置非均匀动态核（**cupy JIT 自定义 CUDA kernel**，硬依赖 cupy）。无工厂函数，直接 `DWNet(dynamic=..., inhomogeneous=..., num_classes=...)` |

### 4.7 作者自研模块

| 文件 | 说明 |
|---|---|
| `Edge_feature_guidance.py` | **EFG_Module 原型**（`EFG_Module(in_channels, reduction=8)`）：特征压到 2 通道提取低维"边缘/方向"表示再升回，与旁路降维特征 concat 后恢复通道——"边缘特征引导"的最早独立定义 |
| `block.py` | 自建即插即用模块库：`SE_Block`、`SEBlock`、**`EFG_Module`**（被 Swin_ET / Vit_ET / swin_efg / Pvt_ET 引用的版本）、`LIG`、`PVT_EFG_Module`、`GhostModule`、`ASPP`、`channel_attention/spatial_attention/CBAM` |

### 4.8 `models/optimize/` —— DEGConv + CPCA 改进主线（Ours 模型所在）

包内全部文件带 `resnet34_base` 基线臂，采用**零初始化 gamma 门控残差**（`x + gamma * module(x)`，初始等价原 ResNet，保护训练稳定）。依赖 einops。

| 文件 | 内容 | 工厂函数 |
|---|---|---|
| `resnet34_CPCA.py` | ResNet34 + **CPCA**（Channel Prior Convolutional Attention）：通道注意力（avg+max 双池化共享 fc）+ 空间注意力（5×5 / 1×7·7×1 / 1×11·11×1 / 1×21·21×1 **多尺度条带深度卷积金字塔**） | `resnet34`（可选 use_cpca）、`resnet34_base`、`resnet34_cpca` |
| `resnet34_CPCA_v1.py` | **DEGConv（方向引导边缘门控卷积）原版 + 改进版 CPCA**。DEGConv：水平/垂直条带深度卷积 → HoG 方向直方图（Sobel 梯度角，8 方向 bin）生成方向嵌入 → sigmoid 边缘门控（作者将逐 cell Python 循环向量化）。限制：只能放 stem/layer1/layer2 | `resnet34_base / resnet34_degconv / resnet34_cpca / resnet34_degconv_cpca / resnet34_degconv_cpca_orig`（CPCA 原版对照臂） |
| `resnet34_CPCA_v2.py` | **DEGConv V2** 四点改进：可微高斯软分箱、空间方向嵌入（保留 cell 网格双线性上采样）、去掉 2×2 patch 切分、自适应 cell 网格（可放任意 stage） | 同上 5 个工厂 |
| `resnet34_msinit.py` | DEGConv + **SpatialAttention**（CBAM 式 7×7 空间注意力替代 CPCA） | `resnet34_base / resnet34_degconv / resnet34_sa / resnet34_degconv_sa` |
| `test.py` | DEGConv+CPCA 组合实现（名为 test，实为组合臂） | `resnet34 / resnet34_base / resnet34_cpca` |
| `__init__.py` | 空包标记 | — |

> 注册表中 `degconv_cpca_v1/v2/v3` 即论文里的 **Ours（ResNet34+DegConv+CPCA）**，由 `a.sh` 训练（img_size 512）。

### 4.9 工具文件

| 文件 | 说明 |
|---|---|
| `Adan.py` | Adan 优化器（ADAptive Nesterov momentum，含 fused 多 tensor 实现），可选 optimizer |
| `Lion.py` | Lion 优化器（符号动量 + 解耦权重衰减），可选 optimizer |
| `transforms.py` | 成对 (image, target) 检测/分割风格数据增强（Compose/RandomResize/RandomCrop 等，移植自 torchvision detection references） |
| `utils.py` | ConvNeXtV2 配套层：LayerNorm（双格式）、GRN；另含依赖 **MinkowskiEngine** 的稀疏张量版层（import 失败会连带 ConvNeXtV2 不可加载） |
| `model_config.py` | `get_config(mode)` 返回 MobileViT 三档结构配置，被 Mobilevit(.py/_ET) 依赖 |

### 4.10 预训练权重加载实况汇总

- **自动下载**（torch.hub）：Deit、Cait、crossvit、PoolFormer
- **本地路径加载**：Pit（`weights/pit_*.pth`）、Cvt
- **定义了 URL 但未启用/被注释**：SqueezeNet、MnasNet、VGGNet、EfficientFormer(V1/V2)、CSWin、Coat、Vig/PVig、TNT、Pvt、Convnext
- **纯从零训练**：其余全部（含所有 `_ET` 改进版与 optimize/ 全部版本）；实际流程中 ImageNet 初始化权重统一放在根目录 `pth/`，由 `train_cle_fair.py --weights` 加载

---

## 5. attention/ 注意力模块库

25 个互相独立的"注意力积木"，供 `models/` 引用做插入实验。目前实际被引用的只有 `CBAM.BasicBlock`（被 Mobilevit_ET / MobileNet / VanilaNet）和 `CoordAtt.CoordAtt`（被 Vig_ET），其余为备选实验模块。

| 文件 | 机制 | 提供的类 |
|---|---|---|
| `AFF.py` | AFF 注意力特征融合 | `DAF`（直接相加基线）、`AFF`（局部+全局注意力加权融合） |
| `BAM.py` | BAM（CVPR2018） | `ChannelGate`、`SpatialGate`、`BAM`（残差式 `1+sigmoid(c*s)`） |
| `CBAM.py` | CBAM（ECCV2018） | `ChannelAttention`、`SpatialAttention`、`BasicBlock`（**内嵌 CA+SA 的残差块，被项目引用最多**；conv3x3 为深度卷积版） |
| `CoordAtt.py` | Coordinate Attention（CVPR2021） | `h_sigmoid`、`h_swish`、`CoordAtt`（H/W 双方向条带池化） |
| `CSELayer.py` | 跨层 SE（接收 dict 输入） | `CSELayer` |
| `cSE.py` | 通道 SE（医学分割 concurrent 系） | `cSE` |
| `DAHead.py` | DANet 双注意力头（简化分割版） | `PositionAttention`、`ChannelAttention`、`DAHead` |
| `DoubleAttention.py` | A2-Net Double Attention（NeurIPS2018） | `DoubleAttentionLayer`（特征收集 bmm + 特征分发） |
| `drloc.py` | DenseRelativeLoc 密集相对位置自监督头（非注意力） | `DenseRelativeLoc` + 采样辅助函数 |
| `DualAttention.py` | DANet 的 PAM/CAM | `PAM_Module`、`CAM_Module` |
| `ECA.py` | ECA-Net（CVPR2020） | `eca_layer`（GAP→一维卷积 k=3，极轻量） |
| `GAMAttention.py` | GAM 全局注意力 | `GAM_Attention`（通道 MLP + 空间 7×7 卷积） |
| `GCBlock.py` | GCNet 全局上下文模块 | `ContextBlock` |
| `GENet.py` | Gather-Excite + ResNeXt 完整骨干 | `GeResNeXt`、`ge_resnext29_8x64d/16x64d` |
| `iAFF.py` | iAFF（AFF 迭代版） | `DAF`、`iAFF`（二次注意力重融合） |
| `MBConv.py` | MBConv 块（含 GELU 版 SE，ConvNeXt 风格） | `MBConv`、`PreNorm`、`FeedForward` 等 |
| `MC-CAM.py` | MS-CAM 多尺度通道注意力 | `MS_CAM`（**文件名含连字符，需 importlib 导入**） |
| `SCConv.py` | SCNet 自校准卷积（CVPR2020，完整骨干） | `SCConv`、`SCBottleneck`、`scnet50/101` 等 |
| `scSE.py` | concurrent Spatial+Channel SE | `sSE`、`cSE`、`csSE` |
| `SE.py` | 原版 SE | `SELayer`（FC 降维 r=16） |
| `ShuffleAttention.py` | SA-Net Shuffle Attention（CVPR2021） | `sa_layer`（通道/空间减半分组 + channel_shuffle） |
| `SKNet.py` | Selective Kernel Networks | `SKConv`（多核分支注意力融合；遗留调试 print） |
| `SplitAttention.py` | ResNeSt Split-Attention | `SplAtConv2d`、`rSoftMax` |
| `sSE.py` | 空间 SE 单独版 | `sSE`（1×1 conv → 单通道 sigmoid 掩码） |
| `TripletAttention.py` | Triplet Attention（ICASSP2021） | `TripletAttention`（H/W/原始三视角 SpatialGate 取平均） |

---

## 6. bow_fusion/ 双流融合框架

**"LBP-BoW 纹理特征 + CNN 深度特征"双流融合实验**。设计原则（见其 [README.md](bow_fusion/README.md)）：不改动 classifyML 任何原始文件，把根目录 fair 训练协议（val macro-F1 选优、run_meta.json、ImageNet 归一化）与 BoW 项目的双流分支融合为一个双分支网络，评估完全复用根目录 `evaluate.py`，保证与单流口径一致可直接横向比较。

```
数据层:  bow_extractor.py ──> my_dataset_bow.py        (img, bow_feat, label)
模型层:  fasternet_cpca.py / fasternet_clecpca.py ──> dual_stream_model.py (DualStreamFusion)
                                        └──> dual_stream_resnet34_cpca.py（锁定主干封装）
训练:    train_dual_fair.py ──> weights/dual_<model>/{model_best.pth, train_log.csv, run_meta.json}
评估:    extract_and_classify_dual.py ──> outputs/ml_classify_dual_<时间戳>/...
```

| 文件 | 行数 | 说明 |
|---|---|---|
| `bow_extractor.py` | 128 | 纹理特征层：多尺度 LBP（skimage，P=8，method='uniform'）直方图，scales=(16,32,64)、K=256 → **768 维 BoW**。`extract_and_cache_bow()` 批量提取并 pkl 缓存；`extract_lbp_from_pil()` 实时提取（align_bow 模式）；`BOWFeatureCache` 懒加载 |
| `my_dataset_bow.py` | 113 | `MyDataSetBOW.__getitem__` 返回 (img, bow_feat, label)。两种 BoW 模式：`align_bow=False` 用全图缓存；`align_bow=True` 单次遍历 transform，对 RandomResizedCrop/**RandomCrop** 记录裁剪区域、对同一裁剪区实时提 BoW（与 CNN 空间严格对齐） |
| `dual_stream_model.py` | 204 | **模型层核心**：`DualStreamFusion`——语义流 backbone（`_strip_classifier` 去头并推断 feat_dim）→ `semantic_proj` 到 256 维；纹理流 BoW → `texture_mlp`；融合用 `nn.MultiheadAttention` 的 Cross-Attention（**Q=纹理，KV=语义**）后与语义 token 拼接 → classifier（置 Identity 时返回 512 维融合特征供 ML 用）。`build_model()` 先查根目录注册表，未命中查本地扩展表 `{fast_t0_cpca, fast_t0_clecpca}` |
| `train_dual_fair.py` | 455 | **训练入口**：继承公平协议（AdamW+Cosine+AMP+label smoothing 可选+val macro-F1 选优+early stop 30）；`_load_backbone_weights` 兼容单骨干预训练（重映射 `backbone.*` 前缀）与双流 checkpoint；`--freeze-backbone` 冻结骨干只训融合头。输出 `weights/dual_<model>[-N]/` |
| `extract_and_classify_dual.py` | 444 | **ML 检查/评估入口**：提取/缓存 BoW → classifier=Identity 作特征提取器 → 归一化自动读 run_meta.json → 特征存 npy → StandardScaler → E2E 基准 + 8 种 ML 分类器（SVM-RBF/Linear、LR、KNN、RF、XGBoost、LightGBM、Voting 软投票），每个方法调用根目录 `evaluate.compute_metrics/save_outputs`。输出 `outputs/ml_classify_dual_<时间戳>/`（每方法 predictions.csv/results.json/evaluation_report.md + comparison_summary.csv） |
| `dual_stream_resnet34_cpca.py` | 93 | 把主干锁定为 `resnet34_degconv_cpca`（DEGConv 浅层+CPCA 深层，feat_dim=512）的命名入口，含前向/反向/特征提取自检 |
| `fasternet_cpca.py` | 234 | 新骨干 **fast_t0_cpca**（feat_dim=1280，约 2.87M）：复用 `models/FasterNet.py` 的 fast_t0 全部子模块（不改原文件），在深层 Stage2/3 出口插入 CPCA（多尺度条带深度卷积空间注意力）；fast_t0 预训练权重可直接载入，CPCA 随机初始化 |
| `fasternet_clecpca.py` | 234 | **CLECPCA 改进版**（约 3.12M）：通道注意力加 std 池化支路；新增 3×3−3×5 深度卷积**高频细节分支**；softmax 自适应多尺度融合（可学习 branch_w）；残差 + proj_out **零初始化**（ReZero 思路，保护预训练特征）；条带核按 stage 深度自适应 |
| `README.md` | — | 目录说明：职责表、依赖、训练/评估命令、更换主干的三种方式、自检命令 |

目录内已有产出：`weights/`（10 个 run）、`outputs/`（2 次 ML 检查）、`bow_cache/`（3 个 pkl 缓存）。

---

## 7. heatMap/ 热力图工具

| 文件 | 说明 |
|---|---|
| `heatMap/utils.py`（229 行） | 自包含 Grad-CAM 实现（改自 jacobgil/pytorch-grad-cam 简化版）：`ActivationsAndGradients`（目标层 hook，支持 `reshape_transform` 把 `[B,N,C]` token 序列变 `[B,C,H,W]`，**从而兼容 ViT/Swin 类模型**）；`GradCAM`（梯度全局平均→通道加权→ReLU→归一化→resize，支持多层目标取平均）；`show_cam_on_image`（JET 叠加）；`center_crop_img` |
| `heatMap/heatMap.py`（73 行） | 使用入口：加载 ViT + 训练权重（**硬编码 Windows 路径**，coffee 数据集），`target_layers=[model.blocks[-1].norm1]`（最后 attention block 的 norm1，因最终分类基于 class token），`ReshapeTransform` 去 cls token reshape 成 14×14，指定 target_category 生成 CAM 并显示 |

机制上支持任意 CNN（reshape_transform=None）与 Transformer 类模型；运行 `python heatMap/heatMap.py` 需先改脚本内路径。

---

## 8. visualization/ 绘图脚本

三个互相独立的一次性画图脚本，数据与路径硬编码，"按需改数重跑"：

| 文件 | 说明 |
|---|---|
| `draw_bar_chart.py` | 分组柱状图（3 类别 × 2 组 Precision 对比，柱顶标数值），保存 `./figure2/12a.jpg`；上半部分为被注释的 7 类 Recall 图代码 |
| `draw_line_chart.py` | 从 txt 读多列数据画折线图（K-Means/GMM 等聚类代价对比，中文 SimHei 字体）；顶部保留 PVT/PVT+EFG 验证 loss 曲线的注释代码 |
| `draw_pr_curves.py` | 用 sklearn `precision_recall_curve` 对硬编码样本画 PR 曲线示例（12 行） |

---

## 9. mobileFormer/ 模型包

timm 风格的自包含 Mobile-Former（CVPR 2022）实现，**项目其它文件不 import 它**（用包内本地 registry）：

| 文件 | 行数 | 说明 |
|---|---|---|
| `MobileFormer.py` | 511 | MobileFormer 主体：可学习全局 token + CNN stem，`features` 为 DnaBlock 序列，末尾 Local2Global + MergeClassifier；`(x, tokens)` 元组在双分支间并行流转。`@register_model` 注册 `mobile_former_{508m,294m,214m,151m,96m,52m,26m}` 7 个变体 |
| `dna_blocks.py` | 1044 | 核心 "DNA" 模块集：`DyReLU`（动态 ReLU 三模式）、`HyperFunc`（全局 token 生成超函数调制 CNN 激活）、`Local2Global`（L2G）、`GlobalBlock`（G2G）、`Global2Local`（G2L）、`DnaBlock(3)`、`MergeClassifier` 等 |
| `helpers.py` | 479 | timm 风格构建/加载工具：`load_pretrained`（HF hub、module. 前缀剥离）、`build_model_with_cfg`、剪枝适配等 |
| `registry.py` | 142 | `@register_model` 装饰器 + `list_models/is_model/model_entrypoint` 等 |

---

## 10. scripts/ 数据与实验辅助脚本

### 10.1 共识消融配套脚本（5 个，2026-09 新增）

| 脚本 | 用途 |
|---|---|
| `build_consensus_ablation_manifests.py` | **manifest 构建器**：从三份样本质量评分 CSV 以 sample_id 集合运算生成 R/V/C/RV/RC/VC 六个训练 manifest（不复制图像，只记录 `sample_id,class_name,relative_path`）+ `MANIFEST_METADATA.json`。内置全部防错：三份输入 sample_id 宇宙必须相同（4977）、与 Full 扫描一致、与 val/test 零重叠、子集计数必须等于预注册期望值（3500/3474/3749/3103/3141/3115）、计算 RVC 与权威 CSV 及现有 HQ 三方双向差集为 0、原子写入、`--verify-only` 只校验 |
| `validate_consensus_ablation_data.py` | **CPU-only 数据验证**（training_started=false）：回归验证 Full（4977，1578/2136/1263，可选有序路径/标签 SHA-256 基线）；校验固定 val=538、test=671、预训练权重 SHA-256；六组 manifest 过滤后实际 DataLoader 读取一个 batch 检查 shape |
| `verify_consensus_run.py` | **单 run 完整性验收器**（批处理每步后调用）：manifest 哈希一致、best checkpoint >1MB、train log 列齐全、run_meta.json 18 个字段逐一等于预注册值；`--require-eval` 时校验 predictions.csv 恰好 671 行、3 列概率有限且 [0,1]、逐行和为 1 |
| `summarize_consensus_ablation.py` | **统一指标汇总**：从 24 份锁定的 predictions.csv（6 旧 Full/RVC + 18 新）用 sklearn 统一重算全部指标（禁止拼接旧 summary CSV），三 seed 以 ddof=1 聚合；产出 METRICS.json、SEED_METRICS.csv、SUMMARY.csv；`--dry-run` 只验证旧文件 |
| `finalize_consensus_ablation.py` | **定稿报告生成器**：读 METRICS.json（要求 24 run 齐全），生成 OVERALL/PER_SEED/PER_CLASS CSV、24 个混淆矩阵 CSV 与叙事分析报告 ANALYSIS.md（含 12 个递进差值、对 10 个预注册问题的程序化回答、结论边界声明） |

### 10.2 早期数据预处理脚本（7 个，硬编码路径的一次性脚本）

| 脚本 | 用途 |
|---|---|
| `create_list.py` | 检测标注 XML 目录随机生成 trainval/train/val/test 四个 txt 列表 |
| `fps.py` | `measure_inference_speed()`：GPU 推理测速（warmup 5 次，输出 FPS 与 ms/图）；尾部有未受保护的测试代码，单独运行会报错 |
| `handle_dataset.py` | 按 8:1:1 把各类图片复制到 train/val/test |
| `read_csv.py` | 读 label CSV，按 0/1/2 把图片移到类别目录 |
| `split_data.py` | 按 10%/10% 抽样复制图片到 val/test（flower） |
| `split_data_v2.py` | 按 10%/10% 随机移动图片到 test/val（wheat） |
| `test.py` | 读 xlsx 标注，按 Multiclass.Label 把图片移到类别文件夹 |

---

## 11. tests/ 单元测试

`tests/test_consensus_ablation.py`（标准库 unittest，服务器无 pytest）：4 个测试类约 10 个用例，覆盖共识消融全部关键防线：

1. **StrictPretrainedLoadingTests**：`load_pretrained_weights_strict` 只允许分类头缺失，主干缺失/多余 key 必须 RuntimeError；vanilla Swin 的 mask buffer 白名单。
2. **ManifestFilterTests**：有效过滤与计数；拒绝重复 sample_id、类别与目录不一致、越界路径（`../`）、与 val/test 泄漏重叠。
3. **SummaryMetricTests**：`compute_metrics`（混淆矩阵/per_class/macro_auroc_ovr）与 `aggregate_seed_metrics`（ddof=1 样本标准差）。
4. **RunMetadataTests**：无 manifest 保持旧 13 字段（向后兼容）；有 manifest 含约 27 个可复现字段。

运行：`python -m unittest discover tests`（或 `python tests/test_consensus_ablation.py`）。

---

## 12. 共识消融实验（manifests / audit / results）

### 12.1 实验设计（详见 `consensus_ablation_audit/CONSENSUS_ABLATION_PLAN.md`）

科学问题：Full vs HQ 只证明 RVC(2880) 优于 Full(4977)，无法区分收益来自单筛选器 / 双共识 / 三模型增量共识 / 数据量正则化 / 类别组成变化。

设计：固定 ResNet34 分类器与全部超参（AdamW、lr 1e-4、wd 0.05、Cosine、AMP、early stop 30、batch 32、seed 42/43/44），唯一变量是训练集合。三个"好样本"筛选器：

| 筛选器 | 来源 | 判据 | 数量 |
|---|---|---|---:|
| **R** | `outputs_3class_resnet_newnormal/train_sample_grading.csv`（KFold.py ResNet34） | grade ∈ {A,B,C} | 3500 |
| **V** | `outputs_3class_vit_newnormal/train_sample_grading.csv`（KFold.py ViT） | grade ∈ {A,B,C} | 3474 |
| **C** | 外部 `../CUFIT0613/cufit_lpm_oof_3class_newnormal.csv`（CUFIT LPM） | correct_rate ≥ 0.5 | 3749 |

八组：Full(4977)、R、V、C、RV(3103)、RC(3141)、VC(3115)、RVC(2880=现有 HQ)。主指标 Macro-F1。另设计了 Random-2880 等规模对照（**尚未运行**，为明确的后续工作）。

### 12.2 `manifests/consensus_ablation/`（7 个文件）

- `R.csv / V.csv / C.csv / RV.csv / RC.csv / VC.csv`：六个训练 manifest（`sample_id,class_name,relative_path`，按 sample_id 排序），是**唯一权威的子集定义**，由 `train_cle_fair.py --train-manifest` 传入，SHA-256 记录进每个 run 的 run_meta.json。
- `MANIFEST_METADATA.json`：生成时间、全部输入 CSV 的路径与 SHA-256、各子集计数与 SHA-256、RVC 三方复核记录（刻意不生成 RVC.csv，RVC 永远复用现有 HQ 数据集）。

### 12.3 `consensus_ablation_audit/`（预注册与审计留痕）

| 文件 | 内容 |
|---|---|
| `CONSENSUS_ABLATION_PLAN.md` | 实验计划书：科学问题、设计、历史协议、集合重算、18 run 训练矩阵、manifest 方案（A 被采纳 / B 软链接被否决）、8 项训练前风险 |
| `CONSENSUS_ABLATION_FILES.md` | 输入文件审计：4 份评分 CSV、数据集、可复用的 6 个旧权重目录、代码入口 |
| `CONSENSUS_ABLATION_COMMANDS.md` | 审计后的执行命令（预检查 → manifest → 训练 → 评估 → 汇总 → Random-2880 设计） |
| `IMPLEMENTATION_CHECK.md` | TDD 记录（RED→GREEN）、无 manifest 向后兼容（有序路径/标签 SHA-256 不变）、CPU dry-run、汇总 dry-run |
| `TRAINING_READY.md` | 训练就绪声明（TRAINING_READY = TRUE，9 项检查） |
| `CANARY_RUN_CHECK.md` | 金丝雀 run（R/seed42，62 epochs 早停，test Macro-F1 0.8793）通过后放行剩余 17 run |
| `LEGACY_PREDICTION_MAP.json` | **锁定复用的 6 份旧预测文件**（Full/RVC × 3 seed，含绝对路径、行数、SHA-256；策略"绝不自动选最新 eval 目录"） |
| `run_consensus_ablation_batch.sh` | **实际执行的总驱动脚本**：6 subset × 3 seed = 18 run 串行（见第 13 节） |
| `CONSENSUS_ABLATION_SUBSET_COUNTS.csv` | 八组训练集类别组成（RVC 相比 Full：Normal 31.7%→45.2%，Malignant 25.4%→16.1%） |
| `MANIFEST_VALIDATION.csv` | 六个 manifest 逐项验证结果（全部 PASS）与 SHA-256 |
| `preimplementation_snapshot/train_cle_fair.py.before_manifest` | 修改前训练脚本快照（回归比对用） |

### 12.4 `consensus_ablation_results/`（最终结果）

| 文件 | 内容 |
|---|---|
| `CONSENSUS_ABLATION_METRICS.json` | 58KB 完整指标库：24 个 run 各自重算指标 + 八组三 seed 聚合 |
| `CONSENSUS_ABLATION_ANALYSIS.md` | 叙事分析报告（finalize 脚本生成） |
| `CONSENSUS_ABLATION_OVERALL.csv` / `PER_SEED.csv` / `PER_CLASS.csv` / `SEED_METRICS.csv` / `SUMMARY.csv` | 各粒度指标表 |
| `CONFUSION_MATRICES/` | 24 个 `<subset>_seed<seed>.csv` |

### 12.5 核心结论（Macro-F1，mean±std）

| Subset | N | Macro-F1 |
|---|---:|---:|
| Full | 4977 | 0.8764 ± 0.0023 |
| R | 3500 | 0.8922 ± 0.0122 |
| V | 3474 | 0.9013 ± 0.0138 |
| C | 3749 | 0.8925 ± 0.0058 |
| RV | 3103 | 0.9053 ± 0.0055 |
| **RC** | **3141** | **0.9124 ± 0.0115（最优）** |
| VC | 3115 | 0.8951 ± 0.0093 |
| RVC | 2880 | 0.9011 ± 0.0098 |

1. 三个单筛选器全部优于 Full（+1.6~+2.5 pp）——"筛掉低质量样本"本身就是主要收益。
2. 双筛选共识 6 个 pairwise 比较中 5 个优于单筛选（唯一反例 VC − V = −0.6 pp）。
3. **三模型共识 RVC 并非最优**（低于 RC 达 1.1 pp），"三模型共识必然更好"不成立。
4. 训练集大小与 Macro-F1 非单调。
5. **Malignant 类受益最明显**：recall 0.833 → 0.84~0.91，F1 0.824 → 0.85~0.885。
6. 边界声明：筛选显著改变类别组成，八组为嵌套/重叠集合未做显著性检验，Random-2880 对照未运行——因果解释需谨慎。

---

## 13. Shell 脚本

| 脚本 | 状态 | 说明 |
|---|---|---|
| `train_all_pretrained.sh` | 旧路径已失效 | **Full vs HQ 主对照入口**：6 模型 × 2 数据集（hq 2880 / full 4977）× 3 seeds = 36 个训练任务，全部调 `train_cle_fair.py`（统一 224 / batch 32 / lr 1e-4 / epochs 300 / early stop 30 / ImageNet 归一化）；断点友好（best 已存在即跳过）。产物 `weights_3class_{hq,full}_seed{42-44}/`。其中 6 个 ResNet34 run 后被共识消融复用为 Full/RVC 组 |
| `train_fair_baseline.sh` | 旧路径已失效 | 二分类公平基线：4 模型 × 3 seeds = 12 runs（`datasets/2class`），输出 `weights_fair/` |
| `kfold.sh` | — | ResNet34 冒烟验证：调 `KFold.py`（3 分类、1 round × 5 fold、50 epochs、early stop monitor=macro_f1）输出 `outputs_smoke_resnet_v2/`；结尾 grep 检查 best_epoch 分布是否健康 |
| `a.sh` | 旧路径已失效 | 训练 **Ours**：`train_cle_fair.py --model resnet34_degconv_cpca_v3 --num_classes 3 --img_size 512 --seed 44`，训练数据为 HQ，加载 ImageNet ResNet34 权重（512 输入为 Ours 专属配置） |
| `b.sh` | 旧路径已失效 | 手工批量评估雏形（后被 `run_eval_all.py` 取代）：seed44 的 6 模型 × full/hq 两组调 `evaluate.py` |
| `consensus_ablation_audit/run_consensus_ablation_batch.sh` | **当前有效路径** | **共识消融总驱动**：串行执行 6 subset × 3 seed = 18 run。每 run：nvidia-smi 记录 + manifest SHA-256 内联校验 → 已完整（verify --require-eval 通过）跳过 / 已训练未评估只补评估 / 不完整残留保留现场并失败退出 → `train_cle_fair.py --model resnet34 --train-manifest manifests/consensus_ablation/<subset>.csv` → `verify_consensus_run.py` 验收 → `evaluate.py` 评估固定 test（输出固定 `eval_fixed_test/`）→ `--require-eval` 验收；全程写 master_training.log（实际 2026-09-16 23:35 至 09-17 00:53 于 RTX 4090 完成，每 run 约 3–5 分钟） |

> 注意：项目历史路径为 `/home/login/cle/...`，现迁移为 `/data/cle/sunfuxiang/...`，上述"旧路径已失效"脚本的硬编码路径需更新后才能复用。

---

## 14. 数据、结果与输出目录

### 14.1 KFold 样本分级输出（共识消融的上游数据源）

| 目录 | 说明 |
|---|---|
| `outputs_3class_resnet_newnormal/` | 3 分类 ResNet34 分级；**`train_sample_grading.csv` 是 R 筛选器输入**（15 列：image_path、patient_id、sample_id、pred_correct_rate、grade A–E 等） |
| `outputs_3class_vit_newnormal/` | 3 分类 ViT 分级；**V 筛选器输入** |
| `outputs_3class_vit/`、`outputs_2class_vit/` | 早期 3 分类 / 2 分类 ViT 分级 |
| `outputs_5class_vit(_newnormal)/`、`outputs_5class_resnet_newnormal/` | 5 分类探索性分级实验 |
| `outputs_smoke_fixed/`、`outputs_smoke_resnet(_v2)/` | KFold 冒烟测试（kfold.sh 调试阶段） |
| `outputs/` | 评估输出（eval_*）与 ML 特征分类输出（ml_classify_*、lbp 相关） |

目录结构模式：`<output_dir>/quality_grading/<模型名>/repeat_r/fold_f/oof_predictions.json` + 根级 `train_sample_grading.csv`、`train_patient_grading_summary.csv`、运行日志。

### 14.2 权重目录

| 目录 | 说明 |
|---|---|
| `weights/` | **Ours 系列与试错产物**：resnet34_degconv_cpca_{v1,v2,v3} 及副本、resnet34_msinit 31 个变体、resnet34_sa、resnet34_tarb 等（a.sh 输出）；另含 bow_fusion 的 dual_* 权重 |
| `weights_3class_full_seed{42,43,44}/`、`weights_3class_hq_seed{42,43,44}/` | **Full vs HQ 主对照**（train_all_pretrained.sh 产出）：每目录 6 模型，`<model>/{best.pth, run_meta.json, train_log CSV, class_indices.json, eval_*/}`；**其中 resnet34 的 6 份 predictions.csv 被 LEGACY_PREDICTION_MAP.json 锁定为消融 Full/RVC 组** |
| `weights_3class_seed42/`、`weights_5class_seed42/` | 早期单 seed 实验（含后来移除的 densenet121） |
| `weights_fair/` | 二分类公平基线（21 个 run） |
| `weights_consensus_ablation/` | **消融 18 个新 run**：`<subset>/<seed>/resnet34/{best.pth, run_meta.json, train_log, eval_fixed_test/}` + train_stdout.log、evaluation.log、master_training.log |
| `pth/` | 7 份 ImageNet 预训练权重：resnet34-333f7ec4.pth（87MB，SHA-256 被全链路反复校验）、vit_base、swin_tiny、pvt_tiny、repvit_m1、fasternet_t0、densenet121 |
| `norm_cache/` | `dataset_norm_cache.json`：数据集 Mean/Std 缓存（fingerprint 失效机制） |

### 14.3 其他

- `__pycache__/`、`models/__pycache__/` 等：Python 字节码缓存，可忽略。
- 数据集本体（`datasets/3class`、`datasets/3class_highquality` 等）不在仓库内，路径散见于各 shell 脚本与审计文档。

---

## 15. 核心数据流

```
【主线一：Full vs HQ 对照】
train_all_pretrained.sh ──> train_cle_fair.py ──> weights_3class_{full,hq}_seed*/
        (36 runs)              │ run_meta.json        │
                                │                      v
                                │            evaluate.py（手动或 b.sh）
                                │                      v
                                │            run_eval_all.py ──> summary_{full,hq,compare}.csv
                                v
a.sh（Ours: resnet34_degconv_cpca_v3, 512 输入）
                                v
                    analysis_3seed_summary.md / analysis_3seed_*.csv

【主线二：共识消融】
KFold.py（ResNet34 / ViT，患者级 K 折 OOF）
   ├─> outputs_3class_resnet_newnormal/train_sample_grading.csv ──> R
   └─> outputs_3class_vit_newnormal/train_sample_grading.csv    ──> V
外部 CUFIT LPM OOF CSV ──> C
        v
scripts/build_consensus_ablation_manifests.py ──> manifests/consensus_ablation/{R,V,C,RV,RC,VC}.csv
        v
scripts/validate_consensus_ablation_data.py（CPU dry-run）──> audit/TRAINING_READY.md
        v
run_consensus_ablation_batch.sh（18 runs）
   ├─ train_cle_fair.py --train-manifest <subset>.csv（consensus_manifest.filter_training_samples 过滤+防泄漏）
   ├─ scripts/verify_consensus_run.py（每步验收）
   └─ evaluate.py ──> eval_fixed_test/predictions.csv
        v
scripts/summarize_consensus_ablation.py（+ audit/LEGACY_PREDICTION_MAP.json 锁定的 6 份旧预测 = 24 run）
        v
scripts/finalize_consensus_ablation.py ──> consensus_ablation_results/（METRICS.json / ANALYSIS.md / CSV×N）

【支线：双流融合】
bow_fusion/train_dual_fair.py ──> weights/dual_*/
bow_fusion/extract_and_classify_dual.py ──> outputs/ml_classify_dual_*/

【支线：传统 ML 对照】
lbp_ml_classify.py（LBP×3 配置 × 3 分类器）
extract_and_classify(_mult).py（深度特征 + E2E）
```

---

## 16. 已知问题与注意事项

1. **路径迁移**：项目从 `/home/login/cle/...` 迁移到 `/data/cle/sunfuxiang/...`，`a.sh`、`b.sh`、`train_all_pretrained.sh`、`train_fair_baseline.sh` 及 `b.sh`/`run_eval_all.py` 日志中的硬编码路径已失效，复用前需更新。
2. **requirements.txt 缺 `scikit-image`**：`lbp_ml_classify.py` 与 `bow_fusion/bow_extractor.py` 实际依赖它；`bow_fusion` 还需 opencv。
3. **特殊依赖**：`models/MogaNet.py` 需 mmcv；`models/DWNet.py` 的非均匀动态卷积需 cupy；`models/utils.py` 的部分层需 MinkowskiEngine（import 失败会连带 ConvNeXtV2 不可用）；`models/GAT.py` 缺 import 属教学摘抄。
4. **文件名连字符**：`attention/MC-CAM.py` 无法常规 import，需 importlib。
5. **调试残留**：`attention/SKNet.py`、`mobileFormer/dna_blocks.py` 等构造函数中有遗留 `print`。
6. **class_indices.json 会被覆盖**：每次调用 `utils.read_split_train/test_data` 都会按当次数据集重写该文件，跨数据集工作时注意核对。
7. **历史遗留结构**：`models/Swin.py` 的 EFG 分支为旧实验硬编码（无开关），旧 Swin+EFG checkpoint 依赖它，**不要改动**；受控对照请用 `Swin_vanilla.py`。
8. **消融实验未完成项**：Random-2880 等规模对照尚未运行，八组比较未做显著性检验——引用"RC 最优"结论时需附带此边界声明。
9. **KFold.py 不保存模型权重**（有意设计，只产分级 CSV）；其断点续训以每折 `oof_predictions.json` 为准。
10. **评估归一化口径**：`evaluate.py` 在无 CLI 与 run_meta 时会兜底 ImageNet 并警告；`evaluate_cle_fair.py` 则直接抛错。为保证可比，优先让训练端产出 run_meta.json 并随权重保存。


## 17. CUFIT Full 对比补充（2026-10-22）

CUFIT Full：Macro-F1 90.43±0.38%，Accuracy 90.31±0.39%，AUROC 97.97±0.03%；HQ + Ours 的三个种子平均 Macro-F1 高 0.619 个百分点，未作显著性检验。主汇总及其总体/每类 CSV 已纳入 CUFIT；HQ 未补跑，仅报告 Full。完整报告见 [cufit_comparison_results/COMPARISON_REPORT.md](cufit_comparison_results/COMPARISON_REPORT.md)。

训练及权重源目录：`../CUFIT0613/experiment_full_vs_ours_20261022/`。原训练代码、数据集、历史模型指标和 `summary_full/hq/compare.csv` 的早期原始记录保留；本次只向 3-seed 分析增加结果。
