# repvit_m1_E2E 评估报告

**类别数**: 3  
**样本总数**: 895  
**Benign_lesions**: 256 张  
**Malignant_lesions**: 210 张  
**Normal**: 429 张  

## 0. 推理时间统计

| 指标 | 数值 |
|------|------|
| 推理总时间 | 1.164s |
| 每张图推理时间 | 1.301 ms |
| 测试集图像总数 | 895 张 |
| repvit_m1特征提取时间/张 | 0.0000 ms |
| 端到端推理时间/张 | 1.3008 ms |

## 1. 多分类核心指标

| 指标 | 值 | 说明 |
|------|-----|------|
| Accuracy | 0.5877 | 正确预测数 / 总样本数 |
| Precision | 0.5838 | Macro-averaged 精确率 |
| Recall / Sensitivity | 0.5887 | Macro-averaged 召回率 |
| F1 Score | 0.5829 | Macro-averaged F1 |
| AUROC | 0.7960 | OvR Macro ROC 曲线下面积 |
| AUROC (weighted) | 0.7860 | OvR Weighted ROC 曲线下面积 |
| AUC-PR | 0.6427 | OvR Macro PR 曲线下面积均值 |
| Balanced Accuracy | 0.5887 | 各类 Recall 均值 |
| Cohen's Kappa | 0.3619 | 一致性系数 |
| MCC | 0.3646 | Matthews 相关系数 |

## 2. 补充总体指标

| 指标 | 值 |
|------|-----|
| Macro Precision | 0.5838 |
| Macro Recall | 0.5887 |
| Macro F1-Score | 0.5829 |
| Weighted Precision | 0.6014 |
| Weighted Recall | 0.5877 |
| Weighted F1 (sklearn) | 0.5910 |
| Paper F1 | 0.5945 |

## 3. 每类详细指标（One-vs-Rest）

### Benign_lesions (support=256)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.4796 |
| Recall (Sensitivity) | 0.5977 |
| Specificity | 0.7402 |
| F1-Score | 0.5322 |
| NPV | 0.8212 |

TP=153  FP=166  TN=473  FN=103

### Malignant_lesions (support=210)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.5942 |
| Recall (Sensitivity) | 0.5857 |
| Specificity | 0.8774 |
| F1-Score | 0.5899 |
| NPV | 0.8735 |

TP=123  FP=84  TN=601  FN=87

### Normal (support=429)

| 指标 | 值 |
|------|-----|
| Precision (PPV) | 0.6775 |
| Recall (Sensitivity) | 0.5828 |
| Specificity | 0.7446 |
| F1-Score | 0.6266 |
| NPV | 0.6597 |

TP=250  FP=119  TN=347  FN=179

## 4. 混淆矩阵

| | Pred: Benign_lesions | Pred: Malignant_lesions | Pred: Normal |
|---|------|------|------|
| True: Benign_lesions | 153 | 20 | 83 |
| True: Malignant_lesions | 51 | 123 | 36 |
| True: Normal | 115 | 64 | 250 |
