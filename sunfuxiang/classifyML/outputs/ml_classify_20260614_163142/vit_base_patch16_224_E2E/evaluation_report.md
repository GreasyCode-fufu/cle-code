# vit_base_patch16_224_E2E 评估报告

**类别数**: 3  
**样本总数**: 895  
**Benign_lesions**: 256 张  
**Malignant_lesions**: 210 张  
**Normal**: 429 张  

## 0. 推理时间统计

| 指标 | 数值 |
|------|------|
| 推理总时间 | 1.638s |
| 每张图推理时间 | 1.830 ms |
| 测试集图像总数 | 895 张 |
| vit_base_patch16_224特征提取时间/张 | 0.0000 ms |
| 端到端推理时间/张 | 1.8305 ms |

## 1. 多分类核心指标

| 指标 | 值 | 说明 |
|------|-----|------|
| Accuracy | 0.7285 | 正确预测数 / 总样本数 |
| Precision | 0.7217 | Macro-averaged 精确率 |
| Recall / Sensitivity | 0.7232 | Macro-averaged 召回率 |
| F1 Score | 0.7220 | Macro-averaged F1 |
| AUROC | 0.8849 | OvR Macro ROC 曲线下面积 |
| AUROC (weighted) | 0.8778 | OvR Weighted ROC 曲线下面积 |
| AUC-PR | 0.8016 | OvR Macro PR 曲线下面积均值 |
| Balanced Accuracy | 0.7232 | 各类 Recall 均值 |
| Cohen's Kappa | 0.5709 | 一致性系数 |
| MCC | 0.5711 | Matthews 相关系数 |

## 2. 补充总体指标

| 指标 | 值 |
|------|-----|
| Macro Precision | 0.7217 |
| Macro Recall | 0.7232 |
| Macro F1-Score | 0.7220 |
| Weighted Precision | 0.7272 |
| Weighted Recall | 0.7285 |
| Weighted F1 (sklearn) | 0.7275 |
| Paper F1 | 0.7278 |

## 3. 每类详细指标（One-vs-Rest）

### Benign_lesions (support=256)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.6736 |
| Recall (Sensitivity) | 0.6289 |
| Specificity | 0.8779 |
| F1-Score | 0.6505 |
| NPV | 0.8552 |

TP=161  FP=78  TN=561  FN=95

### Malignant_lesions (support=210)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.7376 |
| Recall (Sensitivity) | 0.7762 |
| Specificity | 0.9153 |
| F1-Score | 0.7564 |
| NPV | 0.9303 |

TP=163  FP=58  TN=627  FN=47

### Normal (support=429)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.7540 |
| Recall (Sensitivity) | 0.7646 |
| Specificity | 0.7704 |
| F1-Score | 0.7593 |
| NPV | 0.7804 |

TP=328  FP=107  TN=359  FN=101

## 4. 混淆矩阵

| | Pred: Benign_lesions | Pred: Malignant_lesions | Pred: Normal |
|---|------|------|------|
| True: Benign_lesions | 161 | 26 | 69 |
| True: Malignant_lesions | 9 | 163 | 38 |
| True: Normal | 69 | 32 | 328 |
