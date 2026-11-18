# resnet34_E2E 评估报告

**类别数**: 3  
**样本总数**: 895  
**Benign_lesions**: 256 张  
**Malignant_lesions**: 210 张  
**Normal**: 429 张  

## 0. 推理时间统计

| 指标 | 数值 |
|------|------|
| 推理总时间 | 1.025s |
| 每张图推理时间 | 1.146 ms |
| 测试集图像总数 | 895 张 |
| resnet34特征提取时间/张 | 0.0000 ms |
| 端到端推理时间/张 | 1.1456 ms |

## 1. 多分类核心指标

| 指标 | 值 | 说明 |
|------|-----|------|
| Accuracy | 0.7006 | 正确预测数 / 总样本数 |
| Precision | 0.7122 | Macro-averaged 精确率 |
| Recall / Sensitivity | 0.6870 | Macro-averaged 召回率 |
| F1 Score | 0.6923 | Macro-averaged F1 |
| AUROC | 0.8649 | OvR Macro ROC 曲线下面积 |
| AUROC (weighted) | 0.8567 | OvR Weighted ROC 曲线下面积 |
| AUC-PR | 0.7699 | OvR Macro PR 曲线下面积均值 |
| Balanced Accuracy | 0.6870 | 各类 Recall 均值 |
| Cohen's Kappa | 0.5273 | 一致性系数 |
| MCC | 0.5310 | Matthews 相关系数 |

## 2. 补充总体指标

| 指标 | 值 |
|------|-----|
| Macro Precision | 0.7122 |
| Macro Recall | 0.6870 |
| Macro F1-Score | 0.6923 |
| Weighted Precision | 0.7169 |
| Weighted Recall | 0.7006 |
| Weighted F1 (sklearn) | 0.7031 |
| Paper F1 | 0.7087 |

## 3. 每类详细指标（One-vs-Rest）

### Benign_lesions (support=256)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.5701 |
| Recall (Sensitivity) | 0.7148 |
| Specificity | 0.7840 |
| F1-Score | 0.6343 |
| NPV | 0.8728 |

TP=183  FP=138  TN=501  FN=73

### Malignant_lesions (support=210)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.8050 |
| Recall (Sensitivity) | 0.6095 |
| Specificity | 0.9547 |
| F1-Score | 0.6938 |
| NPV | 0.8886 |

TP=128  FP=31  TN=654  FN=82

### Normal (support=429)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.7614 |
| Recall (Sensitivity) | 0.7366 |
| Specificity | 0.7876 |
| F1-Score | 0.7488 |
| NPV | 0.7646 |

TP=316  FP=99  TN=367  FN=113

## 4. 混淆矩阵

| | Pred: Benign_lesions | Pred: Malignant_lesions | Pred: Normal |
|---|------|------|------|
| True: Benign_lesions | 183 | 12 | 61 |
| True: Malignant_lesions | 44 | 128 | 38 |
| True: Normal | 94 | 19 | 316 |
