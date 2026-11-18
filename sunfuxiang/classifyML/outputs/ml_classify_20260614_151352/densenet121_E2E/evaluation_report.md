# densenet121_E2E 评估报告

**类别数**: 3  
**样本总数**: 895  
**Benign_lesions**: 256 张  
**Malignant_lesions**: 210 张  
**Normal**: 429 张  

## 0. 推理时间统计

| 指标 | 数值 |
|------|------|
| 推理总时间 | 1.276s |
| 每张图推理时间 | 1.426 ms |
| 测试集图像总数 | 895 张 |
| densenet121特征提取时间/张 | 0.0000 ms |
| 端到端推理时间/张 | 1.4258 ms |

## 1. 多分类核心指标

| 指标 | 值 | 说明 |
|------|-----|------|
| Accuracy | 0.6983 | 正确预测数 / 总样本数 |
| Precision | 0.7062 | Macro-averaged 精确率 |
| Recall / Sensitivity | 0.6855 | Macro-averaged 召回率 |
| F1 Score | 0.6897 | Macro-averaged F1 |
| AUROC | 0.8582 | OvR Macro ROC 曲线下面积 |
| AUROC (weighted) | 0.8522 | OvR Weighted ROC 曲线下面积 |
| AUC-PR | 0.7577 | OvR Macro PR 曲线下面积均值 |
| Balanced Accuracy | 0.6855 | 各类 Recall 均值 |
| Cohen's Kappa | 0.5248 | 一致性系数 |
| MCC | 0.5282 | Matthews 相关系数 |

## 2. 补充总体指标

| 指标 | 值 |
|------|-----|
| Macro Precision | 0.7062 |
| Macro Recall | 0.6855 |
| Macro F1-Score | 0.6897 |
| Weighted Precision | 0.7140 |
| Weighted Recall | 0.6983 |
| Weighted F1 (sklearn) | 0.7014 |
| Paper F1 | 0.7061 |

## 3. 每类详细指标（One-vs-Rest）

### Benign_lesions (support=256)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.5625 |
| Recall (Sensitivity) | 0.7031 |
| Specificity | 0.7809 |
| F1-Score | 0.6250 |
| NPV | 0.8678 |

TP=180  FP=140  TN=499  FN=76

### Malignant_lesions (support=210)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.7879 |
| Recall (Sensitivity) | 0.6190 |
| Specificity | 0.9489 |
| F1-Score | 0.6933 |
| NPV | 0.8904 |

TP=130  FP=35  TN=650  FN=80

### Normal (support=429)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.7683 |
| Recall (Sensitivity) | 0.7343 |
| Specificity | 0.7961 |
| F1-Score | 0.7509 |
| NPV | 0.7649 |

TP=315  FP=95  TN=371  FN=114

## 4. 混淆矩阵

| | Pred: Benign_lesions | Pred: Malignant_lesions | Pred: Normal |
|---|------|------|------|
| True: Benign_lesions | 180 | 9 | 67 |
| True: Malignant_lesions | 52 | 130 | 28 |
| True: Normal | 88 | 26 | 315 |
