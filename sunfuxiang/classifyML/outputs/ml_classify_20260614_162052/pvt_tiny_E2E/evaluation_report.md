# pvt_tiny_E2E 评估报告

**类别数**: 3  
**样本总数**: 895  
**Benign_lesions**: 256 张  
**Malignant_lesions**: 210 张  
**Normal**: 429 张  

## 0. 推理时间统计

| 指标 | 数值 |
|------|------|
| 推理总时间 | 1.069s |
| 每张图推理时间 | 1.194 ms |
| 测试集图像总数 | 895 张 |
| pvt_tiny特征提取时间/张 | 0.0000 ms |
| 端到端推理时间/张 | 1.1942 ms |

## 1. 多分类核心指标

| 指标 | 值 | 说明 |
|------|-----|------|
| Accuracy | 0.7184 | 正确预测数 / 总样本数 |
| Precision | 0.7094 | Macro-averaged 精确率 |
| Recall / Sensitivity | 0.7315 | Macro-averaged 召回率 |
| F1 Score | 0.7172 | Macro-averaged F1 |
| AUROC | 0.8715 | OvR Macro ROC 曲线下面积 |
| AUROC (weighted) | 0.8642 | OvR Weighted ROC 曲线下面积 |
| AUC-PR | 0.7680 | OvR Macro PR 曲线下面积均值 |
| Balanced Accuracy | 0.7315 | 各类 Recall 均值 |
| Cohen's Kappa | 0.5654 | 一致性系数 |
| MCC | 0.5683 | Matthews 相关系数 |

## 2. 补充总体指标

| 指标 | 值 |
|------|-----|
| Macro Precision | 0.7094 |
| Macro Recall | 0.7315 |
| Macro F1-Score | 0.7172 |
| Weighted Precision | 0.7266 |
| Weighted Recall | 0.7184 |
| Weighted F1 (sklearn) | 0.7192 |
| Paper F1 | 0.7225 |

## 3. 每类详细指标（One-vs-Rest）

### Benign_lesions (support=256)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.6093 |
| Recall (Sensitivity) | 0.6641 |
| Specificity | 0.8294 |
| F1-Score | 0.6355 |
| NPV | 0.8604 |

TP=170  FP=109  TN=530  FN=86

### Malignant_lesions (support=210)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.7184 |
| Recall (Sensitivity) | 0.8381 |
| Specificity | 0.8993 |
| F1-Score | 0.7736 |
| NPV | 0.9477 |

TP=176  FP=69  TN=616  FN=34

### Normal (support=429)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.8005 |
| Recall (Sensitivity) | 0.6923 |
| Specificity | 0.8412 |
| F1-Score | 0.7425 |
| NPV | 0.7481 |

TP=297  FP=74  TN=392  FN=132

## 4. 混淆矩阵

| | Pred: Benign_lesions | Pred: Malignant_lesions | Pred: Normal |
|---|------|------|------|
| True: Benign_lesions | 170 | 30 | 56 |
| True: Malignant_lesions | 16 | 176 | 18 |
| True: Normal | 93 | 39 | 297 |
