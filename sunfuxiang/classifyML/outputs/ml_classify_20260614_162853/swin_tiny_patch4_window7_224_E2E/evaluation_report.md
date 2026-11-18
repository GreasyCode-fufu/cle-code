# swin_tiny_patch4_window7_224_E2E 评估报告

**类别数**: 3  
**样本总数**: 895  
**Benign_lesions**: 256 张  
**Malignant_lesions**: 210 张  
**Normal**: 429 张  

## 0. 推理时间统计

| 指标 | 数值 |
|------|------|
| 推理总时间 | 1.241s |
| 每张图推理时间 | 1.387 ms |
| 测试集图像总数 | 895 张 |
| swin_tiny_patch4_window7_224特征提取时间/张 | 0.0000 ms |
| 端到端推理时间/张 | 1.3869 ms |

## 1. 多分类核心指标

| 指标 | 值 | 说明 |
|------|-----|------|
| Accuracy | 0.7184 | 正确预测数 / 总样本数 |
| Precision | 0.7021 | Macro-averaged 精确率 |
| Recall / Sensitivity | 0.7076 | Macro-averaged 召回率 |
| F1 Score | 0.7034 | Macro-averaged F1 |
| AUROC | 0.8690 | OvR Macro ROC 曲线下面积 |
| AUROC (weighted) | 0.8639 | OvR Weighted ROC 曲线下面积 |
| AUC-PR | 0.7763 | OvR Macro PR 曲线下面积均值 |
| Balanced Accuracy | 0.7076 | 各类 Recall 均值 |
| Cohen's Kappa | 0.5555 | 一致性系数 |
| MCC | 0.5562 | Matthews 相关系数 |

## 2. 补充总体指标

| 指标 | 值 |
|------|-----|
| Macro Precision | 0.7021 |
| Macro Recall | 0.7076 |
| Macro F1-Score | 0.7034 |
| Weighted Precision | 0.7156 |
| Weighted Recall | 0.7184 |
| Weighted F1 (sklearn) | 0.7159 |
| Paper F1 | 0.7170 |

## 3. 每类详细指标（One-vs-Rest）

### Benign_lesions (support=256)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.6300 |
| Recall (Sensitivity) | 0.5586 |
| Specificity | 0.8685 |
| F1-Score | 0.5921 |
| NPV | 0.8308 |

TP=143  FP=84  TN=555  FN=113

### Malignant_lesions (support=210)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.7039 |
| Recall (Sensitivity) | 0.7810 |
| Specificity | 0.8993 |
| F1-Score | 0.7404 |
| NPV | 0.9305 |

TP=164  FP=69  TN=616  FN=46

### Normal (support=429)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.7724 |
| Recall (Sensitivity) | 0.7832 |
| Specificity | 0.7876 |
| F1-Score | 0.7778 |
| NPV | 0.7978 |

TP=336  FP=99  TN=367  FN=93

## 4. 混淆矩阵

| | Pred: Benign_lesions | Pred: Malignant_lesions | Pred: Normal |
|---|------|------|------|
| True: Benign_lesions | 143 | 32 | 81 |
| True: Malignant_lesions | 28 | 164 | 18 |
| True: Normal | 56 | 37 | 336 |
