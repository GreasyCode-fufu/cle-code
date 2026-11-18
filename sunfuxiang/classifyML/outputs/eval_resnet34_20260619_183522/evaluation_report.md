# 分类模型评估报告

**模型**: `resnet34`  
**权重**: `weights_3class_hq_seed42/resnet34/resnet34_3class_hq_seed42_best.pth`  
**测试集**: `/home/login/cle/sunfuxiang/datasets/3class/test`  
**样本总数**: 671  
**Benign_lesions**: 256 张  
**Malignant_lesions**: 210 张  
**body_of_stomach_normal**: 205 张  

## 1. 论文核心指标

| 指标 | 点估计 | 95% CI | 说明 |
|------|--------|--------|------|
| AUROC (Macro OvR) | 0.9811 | [0.9739, 0.9872] | macro average, One-vs-Rest |
| Precision (Weighted) | 0.9070 | [0.8853, 0.9267] | precision_score(average='weighted') |
| Accuracy | 0.9031 | [0.8793, 0.9240] | 正确预测数 / 总样本数 |
| Sensitivity / Recall (Weighted) | 0.9031 | [0.8793, 0.9240] | recall_score(average='weighted') |
| F1 Score (Paper) | 0.9051 | [0.8829, 0.9248] | 2*P*R/(P+R), P/R 均为 weighted |

## 2. 模型复杂度与推理效率

### 模型复杂度

| 指标 | 值 | 说明 |
|------|-----|------|
| MACs | 3.68G | 乘加操作数 (thop profile) |
| FLOPs | 7.36G | 1 MAC = 2 FLOPs |
| Params | 21.29M | 可学习参数总量 |

### 推理速度

| 指标 | 值 | 说明 |
|------|-----|------|
| Pure Inference FPS | 3725.38 | 样本数 / 纯前向传播耗时 |
| Pure Inference Time | 0.180s | 模型前向传播总耗时 |
| End-to-End FPS | 245.70 | 样本数 / 端到端总耗时 (含数据加载) |
| End-to-End Time | 2.731s | 含数据加载与预处理的端到端耗时 |
| Total Samples | 671 | 测试集样本总数 |

## 3. 补充总体指标

| 指标 | 点估计 | 95% CI |
|------|--------|--------|
| Balanced Accuracy | 0.9091 | [0.8871, 0.9280] |
| Macro Precision | 0.9043 | [0.8816, 0.9241] |
| Macro Recall (Sensitivity) | 0.9091 | [0.8871, 0.9280] |
| Macro F1-Score | 0.9044 | [0.8807, 0.9246] |
| Micro Precision | 0.9031 | [nan, nan] |
| Micro Recall | 0.9031 | [nan, nan] |
| Micro F1-Score | 0.9031 | [nan, nan] |
| Weighted F1 (sklearn) | 0.9027 | [nan, nan] |
| Cohen's Kappa | 0.8547 | [0.8187, 0.8858] |
| MCC | 0.8570 | [0.8228, 0.8869] |
| AUC-PR (Macro OvR) | 0.9659 | [nan, nan] |

## 4. 每类详细指标（One-vs-Rest）

### Benign_lesions (support=256)

| 指标 | 值 | 95% CI |
|------|-----|--------|
| Precision (PPV) | 0.9464 | — |
| Recall (Sensitivity) | 0.8281 | [0.7798, 0.8730] |
| Specificity | 0.9711 | [0.9536, 0.9858] |
| F1-Score | 0.8833 | [0.8490, 0.9116] |
| NPV | 0.9016 | — |
| AUC-ROC (OvR) | 0.9749 | — |
| AUC-PR (OvR) | 0.9660 | — |

TP=212  FP=12  TN=403  FN=44

### Malignant_lesions (support=210)

| 指标 | 值 | 95% CI |
|------|-----|--------|
| Precision (PPV) | 0.8412 | — |
| Recall (Sensitivity) | 0.9333 | [0.8990, 0.9655] |
| Specificity | 0.9197 | [0.8934, 0.9429] |
| F1-Score | 0.8849 | [0.8515, 0.9142] |
| NPV | 0.9680 | — |
| AUC-ROC (OvR) | 0.9730 | — |
| AUC-PR (OvR) | 0.9424 | — |

TP=196  FP=37  TN=424  FN=14

### body_of_stomach_normal (support=205)

| 指标 | 值 | 95% CI |
|------|-----|--------|
| Precision (PPV) | 0.9252 | — |
| Recall (Sensitivity) | 0.9659 | [0.9385, 0.9894] |
| Specificity | 0.9657 | [0.9471, 0.9813] |
| F1-Score | 0.9451 | [0.9197, 0.9654] |
| NPV | 0.9847 | — |
| AUC-ROC (OvR) | 0.9955 | — |
| AUC-PR (OvR) | 0.9892 | — |

TP=198  FP=16  TN=450  FN=7

## 5. 混淆矩阵

| | Pred: Benign_lesions | Pred: Malignant_lesions | Pred: body_of_stomach_normal |
|---|------|------|------|
| True: Benign_lesions | 212 | 33 | 11 |
| True: Malignant_lesions | 9 | 196 | 5 |
| True: body_of_stomach_normal | 3 | 4 | 198 |

## 6. Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9464    0.8281    0.8833       256
     Malignant_lesions     0.8412    0.9333    0.8849       210
body_of_stomach_normal     0.9252    0.9659    0.9451       205

              accuracy                         0.9031       671
             macro avg     0.9043    0.9091    0.9044       671
          weighted avg     0.9070    0.9031    0.9027       671

```
