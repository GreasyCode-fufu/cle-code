# ClassifyML 使用指南

## 环境配置

```bash
# 创建 conda 环境（如需从头搭建）
conda create -n pytorch-2.1.1 python=3.11 -y
conda activate pytorch-2.1.1

# 安装依赖
pip install -r requirements.txt
```

当前项目使用 conda 环境 `pytorch-2.1.1`（Python 3.11, CUDA 支持）。

---

## 一、数据准备

### 1.1 数据集目录结构要求

所有脚本要求图片按如下结构组织：

```
dataset_root/
├── train/
│   ├── class_a/
│   │   ├── img001.jpg
│   │   └── ...
│   └── class_b/
│       ├── img100.jpg
│       └── ...
├── val/
│   ├── class_a/
│   └── class_b/
└── test/
    ├── class_a/
    └── class_b/
```

### 1.2 数据集划分脚本（scripts/）

以下脚本用于数据预处理，**需要根据实际路径修改脚本内的硬编码路径**后运行。

| 脚本 | 用途 |
|------|------|
| `scripts/split_data.py` | 按比例划分数据集（train/val/test = 80/10/10） |
| `scripts/split_data_v2.py` | 根据 Excel 标注将图片分入类别文件夹 |
| `scripts/handle_dataset.py` | 按比例划分数据集（train/val/test = 60/20/20） |
| `scripts/test.py` | 从训练集中随机抽取样本移入 test/val |
| `scripts/create_list.py` | 生成检测数据集的 train/val/test 列表文件 |
| `scripts/read_csv.py` | 根据 CSV 标签将图片移动到对应类别文件夹 |

```bash
# 示例：修改脚本内路径后运行
python scripts/split_data.py
```

---

## 二、模型训练

### 2.1 端到端训练（train.py）

```bash
python train.py \
  --num_classes 2 \
  --data-path-train /path/to/train \
  --data-path-val /path/to/val \
  --img_size 224 \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-4 \
  --early-stop 20 \
  --device cuda:0 \
  --save-dir ./weights \
  --save-name resnet34_2class.pth \
  --csv-name resnet34_2class.csv
```

**Mean / Std 自动计算：** 训练时会自动从**训练集**计算逐通道 Mean/Std，用于 `transforms.Normalize`。计算结果按数据集 fingerprint（SHA256: 路径+大小+mtime）缓存到 JSON 文件，下次训练相同数据集时直接复用，无需重算。首次运行会打印如下信息：

```
[0/6] 自动获取训练集 Mean / Std ...
  Mean/Std source: training set only
  Train path:      /path/to/train
  Train images:    1234
  img_size:        224
  Norm transform:  Resize(256) + CenterCrop(224) + ToTensor()
  Cache:           miss (缓存未命中，已计算并写入)
  MEAN = [0.2939, 0.2939, 0.2939]
  STD  = [0.2064, 0.2064, 0.2064]
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_classes` | int | 3 | 分类类别数 |
| `--data-path-train` | str | — | 训练集目录（按类别分子文件夹） |
| `--data-path-val` | str | — | 验证集目录 |
| `--img_size` | int | 512 | 输入图像尺寸 |
| `--epochs` | int | 300 | 训练轮数 |
| `--batch-size` | int | 32 | 批量大小 |
| `--lr` | float | 1e-4 | 学习率 |
| `--weights` | str | '' | 预训练权重路径（为空则从头训练） |
| `--freeze-layers` | flag | False | 冻结除分类头外的所有层 |
| `--early-stop` | int | 0 | 早停耐心值（0 表示关闭） |
| `--device` | str | cuda:0 | 训练设备 |
| `--save-dir` | str | ./weights | 权重保存目录 |
| `--save-name` | str | resnet34_cle.pth | 权重文件名 |
| `--csv-name` | str | resnet34_cle.csv | 训练日志文件名 |
| `--recompute-norm` | flag | False | 强制重新计算 Mean/Std，忽略已有缓存 |
| `--norm-cache-dir` | str | ./norm_cache | Mean/Std 缓存目录 |

**切换模型：** 修改 `train.py` 中的 `import` 行即可，例如：
```python
# from models.Resnet import resnet34 as create_model
from models.Mobilevit import mobile_vit_xx_small as create_model
```

**输出：**
- `{save-dir}/{model_name}/resnet34_2class.pth` — 最佳模型权重
- `{save-dir}/{model_name}/resnet34_2class.csv` — 训练日志（epoch, train_loss, train_acc, val_loss, val_acc, lr）

### 2.2 K-Fold 交叉验证训练（KFold.py）

按患者分组进行 K-Fold 交叉验证，同时对样本进行质量分级（A/B/C/D/E）。

```bash
python KFold.py \
  --csv_path /path/to/sample_mapping.csv \
  --data_path_train /path/to/train \
  --num_classes 2 \
  --num_folds 5 \
  --num_rounds 3 \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-4 \
  --img_size 224 \
  --device cuda:0 \
  --output_dir ./outputs/kfold_output
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--csv_path` | str | — | 样本-患者映射 CSV 路径 |
| `--data_path_train` | str | — | 训练数据根目录 |
| `--num_classes` | int | 2 | 分类类别数 |
| `--num_folds` | int | 5 | K-Fold 折数 |
| `--num_rounds` | int | 3 | 重复训练轮数 |
| `--epochs` | int | 100 | 每折训练轮数 |
| `--batch_size` | int | 16 | 批量大小 |
| `--lr` | float | 1e-4 | 学习率 |
| `--img_size` | int | 224 | 图像尺寸 |
| `--base_seed` | int | 42 | 随机种子 |
| `--model_module` | str | models.Resnet | 模型模块路径 |
| `--model_function` | str | resnet34 | 模型创建函数名 |
| `--weights` | str | '' | 预训练权重路径 |
| `--freeze_layers` | flag | False | 冻结层 |
| `--use_early_stop` | flag | False | 启用早停 |
| `--early_stop_patience` | int | 20 | 早停耐心值 |
| `--early_stop_monitor` | str | loss | 监控指标（loss/acc） |
| `--inner_val_ratio` | float | 0.1 | 内部验证集比例 |
| `--output_dir` | str | ./outputs/kfold_output | 输出目录 |
| `--grade_a_acc` | float | 0.95 | A 级准确率阈值 |
| `--grade_b_acc` | float | 0.85 | B 级准确率阈值 |

**输出：**
- `{output_dir}/` — 每折每轮的 OOF 预测
- `sample_grading.csv` — 样本质量分级结果
- `patient_summary.csv` — 患者级别汇总

---

## 三、模型评估

### 3.1 多分类评估（evaluate.py）

适用于 3 分类（benign_change / neoplastic / normal）评估。

```bash
python evaluate.py \
  --model resnet34 \
  --weights ./weights/resnet34/resnet34_cle.pth \
  --num-classes 3 \
  --data-path /path/to/test \
  --class-names benign_change neoplastic normal \
  --img-size 512 \
  --norm-mean 0.2431 0.2431 0.2431 \
  --norm-std 0.2339 0.2339 0.2339 \
  --batch-size 32 \
  --device cuda:0 \
  --n-bootstrap 1000
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | resnet34 | 模型名称（见下方支持列表） |
| `--weights` | str | — | 模型权重路径 |
| `--num-classes` | int | 3 | 分类类别数 |
| `--data-path` | str | — | 测试集目录 |
| `--class-names` | list | — | 类别名称列表 |
| `--img-size` | int | 512 | 输入图像尺寸 |
| `--norm-mean` | list | [0.2431,...] | 归一化均值 |
| `--norm-std` | list | [0.2339,...] | 归一化标准差 |
| `--batch-size` | int | 32 | 推理批量大小 |
| `--device` | str | cuda:0 | 推理设备 |
| `--n-bootstrap` | int | 1000 | Bootstrap 置信区间采样数 |

**支持的模型：** resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d, repvit_m1/m2/m3, fast_t0, vanillanet_5, alexnet, convnext_tiny/small, efficientnet_b0, mobilenet_v2, swin_tiny_patch4_window7_224, vit_base_patch16_224

**输出：**
- `predictions.csv` — 预测结果
- `evaluation_report.md` — 完整评估报告
- `confusion_matrix.png` — 混淆矩阵
- `roc_curve.png` — ROC 曲线
- `pr_curve.png` — PR 曲线
- `per_class_metrics.png` — 每类指标柱状图

### 3.2 二分类评估（evaluate_2class.py）

适用于 2 分类（abnormal / normal）评估。

```bash
python evaluate_2class.py \
  --model resnet34 \
  --weights ./weights/resnet34-2/resnet34_2class.pth \
  --data-path /path/to/test \
  --class-names abnormal normal \
  --img-size 224 \
  --norm-mean 0.2939 0.2939 0.2939 \
  --norm-std 0.2064 0.2064 0.2064 \
  --batch-size 32 \
  --device cuda:0
```

**参数说明：** 与 `evaluate.py` 类似，但固定 `num_classes=2`，`class-names` 默认为 `["abnormal", "normal"]`。

**输出：** 同 `evaluate.py`。

---

## 四、特征提取 + 机器学习分类（extract_and_classify.py）

加载训练好的 ResNet34，去掉 FC 层提取 512 维特征，使用 9 种 ML 分类器进行对比实验。

```bash
python extract_and_classify.py \
  --weights ./weights/resnet34-2/resnet34_2class.pth \
  --data-train /path/to/trainABC \
  --data-test /path/to/test \
  --img_size 224 \
  --batch-size 64 \
  --device cuda:0 \
  --output-dir ./outputs
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | str | 必填 | 训练好的 ResNet 权重路径 |
| `--data-train` | str | 必填 | 训练集目录（用于提取特征训练 ML 分类器） |
| `--data-test` | str | 必填 | 测试集目录 |
| `--img_size` | int | 224 | 输入图像尺寸 |
| `--batch-size` | int | 64 | 推理批量大小 |
| `--device` | str | cuda:0 | 推理设备 |
| `--output-dir` | str | ./outputs | 输出根目录 |

**包含的分类方法（9 种）：**

| 方法 | 子目录名 | 配置 |
|------|----------|------|
| ResNet 端到端 | `ResNet_E2E/` | Softmax 直接分类 |
| SVM (RBF) | `SVM_RBF/` | kernel='rbf', C=1.0, gamma='scale' |
| SVM (Linear) | `SVM_Linear/` | kernel='linear', C=1.0 |
| Logistic Regression | `LogisticRegression/` | max_iter=1000 |
| KNN | `KNN/` | n_neighbors=5 |
| Random Forest | `RandomForest/` | n_estimators=200, max_depth=8 |
| XGBoost | `XGBoost/` | n_estimators=200, max_depth=6, lr=0.1 |
| LightGBM | `LightGBM/` | n_estimators=200, max_depth=6, lr=0.1 |
| Voting Ensemble | `VotingEnsemble/` | 软投票：SVM-RBF + XGBoost + RF |

**输出目录结构：**

```
outputs/ml_classify_{timestamp}/
├── ResNet_E2E/
│   ├── predictions.csv
│   └── evaluation_report.md
├── SVM_RBF/
│   ├── predictions.csv
│   └── evaluation_report.md
├── SVM_Linear/
│   └── ...
├── LogisticRegression/
│   └── ...
├── KNN/
│   └── ...
├── RandomForest/
│   └── ...
├── XGBoost/
│   └── ...
├── LightGBM/
│   └── ...
├── VotingEnsemble/
│   ├── predictions.csv
│   └── evaluation_report.md
├── train_features.npy          # 训练集特征 [N_train, 512]
├── test_features.npy           # 测试集特征 [N_test, 512]
└── comparison_summary.csv      # 所有方法指标汇总对比
```

---

## 四-B、特征提取 + 机器学习多分类（extract_and_classify_mult.py）

`extract_and_classify_mult.py` 是 `extract_and_classify.py` 的多分类扩展版本，支持任意 N 分类任务。加载训练好的模型，去掉分类头提取特征，使用 8 种 ML 分类器进行对比实验。

```bash
python extract_and_classify_mult.py \
  --weights ./weights/fast_t0/fast_t0_cle.pth \
  --data-train /path/to/train \
  --data-test /path/to/test \
  --img_size 512 \
  --batch-size 64 \
  --device cuda:0 \
  --output-dir ./outputs
```

**Mean / Std 自动计算：** 与 `train.py` 一致，自动从**训练集**计算 Mean/Std，结果按 fingerprint 缓存。`extract_and_classify_mult.py` 和 `train.py` 共享同一套缓存机制，只要训练集路径和 `img_size` 相同，即可复用训练时已计算的缓存，无需重算。首次运行会打印：

```
Step 2: 加载数据集
  Mean/Std source: training set only
  Train path:      /path/to/train
  Train images:    1234
  img_size:        512
  Norm transform:  Resize(585) + CenterCrop(512) + ToTensor()
  Cache:           hit (缓存命中)
  MEAN = [0.2939, 0.2939, 0.2939]
  STD  = [0.2064, 0.2064, 0.2064]
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--weights` | str | 必填 | 训练好的模型权重路径 |
| `--data-train` | str | 必填 | 训练集目录（用于提取特征训练 ML 分类器） |
| `--data-test` | str | 必填 | 测试集目录 |
| `--img_size` | int | 224 | 输入图像尺寸（须与训练时的 `--img_size` 一致） |
| `--batch-size` | int | 64 | 推理批量大小 |
| `--device` | str | cuda:0 | 推理设备 |
| `--output-dir` | str | ./outputs | 输出根目录 |
| `--class-names` | list | None | 类别名称列表（空格分隔），未提供则从训练集目录自动检测 |
| `--num-classes` | int | None | 类别数量，用于验证（须与自动检测或 `--class-names` 一致） |
| `--recompute-norm` | flag | False | 强制重新计算 Mean/Std，忽略已有缓存 |
| `--norm-cache-dir` | str | ./norm_cache | Mean/Std 缓存目录 |

**包含的分类方法（8 种）：**

| 方法 | 子目录名 | 配置 |
|------|----------|------|
| 端到端模型 | `{MODEL_NAME}_E2E/` | Softmax 直接分类 |
| SVM (RBF) | `SVM_RBF/` | kernel='rbf', C=1.0, gamma='scale' |
| SVM (Linear) | `SVM_Linear/` | kernel='linear', C=1.0 |
| Logistic Regression | `LogisticRegression/` | max_iter=1000 |
| KNN | `KNN/` | n_neighbors=5 |
| Random Forest | `RandomForest/` | n_estimators=200, max_depth=8 |
| XGBoost | `XGBoost/` | n_estimators=200, max_depth=6, lr=0.1 |
| LightGBM | `LightGBM/` | n_estimators=200, max_depth=6, lr=0.1 |

**输出目录结构：**

```
outputs/ml_classify_{timestamp}/
├── {MODEL_NAME}_E2E/
│   ├── predictions.csv
│   └── evaluation_report.md
├── SVM_RBF/
├── SVM_Linear/
├── LogisticRegression/
├── KNN/
├── RandomForest/
├── XGBoost/
├── LightGBM/
├── train_features.npy              # 训练集特征
├── test_features.npy               # 测试集特征
├── comparison_summary.csv          # 所有方法指标汇总对比
└── per_class_comparison.csv        # 各类别指标对比
```

---

## 五、LBP 特征提取 + 机器学习分类（lbp_ml_classify.py）

基于 LBP（Local Binary Pattern）手工特征 + 传统 ML 分类器，不依赖深度学习模型。支持 3 种 LBP 配置（10/54/486 维），每种配置使用 3 种分类器进行对比。

```bash
python lbp_ml_classify.py \
  --data-train /path/to/trainABC \
  --data-test /path/to/test \
  --img-size 224 \
  --output-dir ./lbp_ml_results
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data-train` | str | 必填 | 训练集目录 |
| `--data-test` | str | 必填 | 测试集目录 |
| `--img-size` | int | 224 | 图像缩放尺寸 |
| `--output-dir` | str | ./lbp_ml_results | 输出根目录 |

**LBP 特征配置：**

| 配置名 | 维度 | 说明 |
|--------|------|------|
| LBP-10 | 10 | 单尺度 P=8,R=1，全图直方图 |
| LBP-54 | 54 | 3 尺度 (8,1)+(16,2)+(24,3)，全图拼接 |
| LBP-486 | 486 | 3×3 分块 + 3 尺度多尺度，可类比 ResNet 512 维 |

**每种配置使用的分类器：**

| 方法 | 配置 |
|------|------|
| Logistic Regression | max_iter=1000, random_state=42 |
| SVM (Linear) | kernel='linear', probability=True, random_state=42 |
| Random Forest | n_estimators=200, max_depth=8, random_state=42 |

**输出目录结构：**

```
lbp_ml_results/lbp_ml_{timestamp}/
├── LBP-10/
│   ├── LogisticRegression/ (predictions.csv, report.md)
│   ├── SVM_Linear/
│   ├── RandomForest/
│   ├── train_features.npy
│   └── test_features.npy
├── LBP-54/
│   └── ...
├── LBP-486/
│   └── ...
└── all_configs_comparison.csv      # 跨配置 × 方法的指标汇总
```

---

## 六、可视化工具

### 6.1 热力图（heatMap/heatMap.py）

使用 Grad-CAM 生成模型注意力热力图。**需修改脚本内硬编码的模型路径、权重路径和图片路径。**

```bash
python heatMap/heatMap.py
```

### 6.2 图表绘制（visualization/）

| 脚本 | 用途 |
|------|------|
| `visualization/draw_bar_chart.py` | 绘制指标对比柱状图 |
| `visualization/draw_line_chart.py` | 绘制聚类代价折线图 |
| `visualization/draw_pr_curves.py` | 绘制 Precision-Recall 曲线 |

以上脚本数据硬编码在脚本内，需根据需要修改后运行。

---

## 七、典型工作流程

### 7.1 三分类任务（CLE 内镜图像）

```bash
# 1. 训练模型（3 分类，Mean/Std 自动从训练集计算并缓存）
python train.py \
  --num_classes 3 \
  --data-path-train /path/to/filtered_abcd/train \
  --data-path-val /path/to/filtered_abcd/val \
  --img_size 512 --epochs 300 --batch-size 32 --lr 1e-4 \
  --device cuda:0

# 2. 评估
python evaluate.py \
  --model resnet34 \
  --weights ./weights/resnet34/resnet34_cle.pth \
  --data-path /path/to/filtered_abcd/test \
  --class-names benign_change neoplastic normal \
  --img-size 512

# 3. 特征提取 + 8 种 ML 多分类器对比（Mean/Std 复用训练时的缓存）
python extract_and_classify_mult.py \
  --weights ./weights/fast_t0/fast_t0_cle.pth \
  --data-train /path/to/filtered_abcd/train \
  --data-test /path/to/filtered_abcd/test \
  --img_size 512 --batch-size 64 --device cuda:0
```

### 7.2 二分类任务 + KFold 样本筛选 + ML 分类

```bash
# 1. KFold 交叉验证 + 样本分级（可选，用于筛选高质量训练数据）
python KFold.py \
  --csv_path /path/to/mapping.csv \
  --data_path_train /path/to/2classCLEall/train \
  --num_classes 2 --num_folds 5 --num_rounds 3 \
  --epochs 100 --img_size 224 --device cuda:0

# 2. 使用筛选后的 trainABC 数据训练 ResNet34（二分类）
python train.py \
  --num_classes 2 \
  --data-path-train /path/to/2classCLEall/trainABC \
  --data-path-val /path/to/2classCLEall/val \
  --img_size 224 --epochs 100 --batch-size 64 --lr 1e-4 \
  --early-stop 20 --device cuda:0 \
  --save-name resnet34_2class.pth --csv-name resnet34_2class.csv

# 3. 端到端评估
python evaluate_2class.py \
  --model resnet34 \
  --weights ./weights/resnet34-2/resnet34_2class.pth \
  --data-path /path/to/2classCLEall/test \
  --class-names abnormal normal --img-size 224

# 4. ResNet 特征提取 + 9 种 ML 分类器对比
python extract_and_classify.py \
  --weights ./weights/resnet34-2/resnet34_2class.pth \
  --data-train /path/to/2classCLEall/trainABC \
  --data-test /path/to/2classCLEall/test \
  --img_size 224 --batch-size 64 --device cuda:0

# 5. LBP 手工特征 + 3 种配置 × 3 种 ML 分类器对比
python lbp_ml_classify.py \
  --data-train /path/to/2classCLEall/trainABC \
  --data-test /path/to/2classCLEall/test \
  --img-size 224
```
