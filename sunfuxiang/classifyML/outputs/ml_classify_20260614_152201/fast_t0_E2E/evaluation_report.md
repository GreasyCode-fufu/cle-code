# fast_t0_E2E 评估报告

**类别数**: 3  
**样本总数**: 895  
**Benign_lesions**: 256 张  
**Malignant_lesions**: 210 张  
**Normal**: 429 张  

## 0. 推理时间统计

| 指标 | 数值 |
|------|------|
| 推理总时间 | 1.121s |
| 每张图推理时间 | 1.252 ms |
| 测试集图像总数 | 895 张 |
| fast_t0特征提取时间/张 | 0.0000 ms |
| 端到端推理时间/张 | 1.2522 ms |

## 1. 多分类核心指标

| 指标 | 值 | 说明 |
|------|-----|------|
| Accuracy | 0.6715 | 正确预测数 / 总样本数 |
| Precision | 0.6781 | Macro-averaged 精确率 |
| Recall / Sensitivity | 0.6588 | Macro-averaged 召回率 |
| F1 Score | 0.6612 | Macro-averaged F1 |
| AUROC | 0.8586 | OvR Macro ROC 曲线下面积 |
| AUROC (weighted) | 0.8540 | OvR Weighted ROC 曲线下面积 |
| AUC-PR | 0.7518 | OvR Macro PR 曲线下面积均值 |
| Balanced Accuracy | 0.6588 | 各类 Recall 均值 |
| Cohen's Kappa | 0.4862 | 一致性系数 |
| MCC | 0.4910 | Matthews 相关系数 |

## 2. 补充总体指标

| 指标 | 值 |
|------|-----|
| Macro Precision | 0.6781 |
| Macro Recall | 0.6588 |
| Macro F1-Score | 0.6612 |
| Weighted Precision | 0.6939 |
| Weighted Recall | 0.6715 |
| Weighted F1 (sklearn) | 0.6766 |
| Paper F1 | 0.6825 |

## 3. 每类详细指标（One-vs-Rest）

### Benign_lesions (support=256)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.5148 |
| Recall (Sensitivity) | 0.6797 |
| Specificity | 0.7433 |
| F1-Score | 0.5859 |
| NPV | 0.8528 |

TP=174  FP=164  TN=475  FN=82

### Malignant_lesions (support=210)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.7425 |
| Recall (Sensitivity) | 0.5905 |
| Specificity | 0.9372 |
| F1-Score | 0.6578 |
| NPV | 0.8819 |

TP=124  FP=43  TN=642  FN=86

### Normal (support=429)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.7769 |
| Recall (Sensitivity) | 0.7063 |
| Specificity | 0.8133 |
| F1-Score | 0.7399 |
| NPV | 0.7505 |

TP=303  FP=87  TN=379  FN=126

## 4. 混淆矩阵

| | Pred: Benign_lesions | Pred: Malignant_lesions | Pred: Normal |
|---|------|------|------|
| True: Benign_lesions | 174 | 21 | 61 |
| True: Malignant_lesions | 60 | 124 | 26 |
| True: Normal | 104 | 22 | 303 |
