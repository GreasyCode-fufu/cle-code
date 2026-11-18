# 分类模型评估报告

**模型**: `resnet34`  
**权重**: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed42/resnet34/resnet34_3class_full_seed42_best.pth`  
**测试集**: `/home/login/cle/sunfuxiang/datasets/3class/test`  
**样本总数**: 671  
**Benign_lesions**: 256 张  
**Malignant_lesions**: 210 张  
**body_of_stomach_normal**: 205 张  

## 1. 论文核心指标

| 指标 | 点估计 | 95% CI | 说明 |
|------|--------|--------|------|
| AUROC (Macro OvR) | 0.9643 | [0.9536, 0.9739] | macro average, One-vs-Rest |
| Precision (Weighted) | 0.8760 | [0.8503, 0.9014] | precision_score(average='weighted') |
| Accuracy | 0.8763 | [0.8510, 0.9016] | 正确预测数 / 总样本数 |
| Sensitivity / Recall (Weighted) | 0.8763 | [0.8510, 0.9016] | recall_score(average='weighted') |
| F1 Score (Paper) | 0.8761 | [0.8503, 0.9012] | 2*P*R/(P+R), P/R 均为 weighted |

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
| Pure Inference FPS | 3744.38 | 样本数 / 纯前向传播耗时 |
| Pure Inference Time | 0.179s | 模型前向传播总耗时 |
| End-to-End FPS | 258.72 | 样本数 / 端到端总耗时 (含数据加载) |
| End-to-End Time | 2.594s | 含数据加载与预处理的端到端耗时 |
| Total Samples | 671 | 测试集样本总数 |

## 3. 补充总体指标

| 指标 | 点估计 | 95% CI |
|------|--------|--------|
| Balanced Accuracy | 0.8785 | [0.8544, 0.9025] |
| Macro Precision | 0.8778 | [0.8526, 0.9023] |
| Macro Recall (Sensitivity) | 0.8785 | [0.8544, 0.9025] |
| Macro F1-Score | 0.8781 | [0.8536, 0.9022] |
| Micro Precision | 0.8763 | [nan, nan] |
| Micro Recall | 0.8763 | [nan, nan] |
| Micro F1-Score | 0.8763 | [nan, nan] |
| Weighted F1 (sklearn) | 0.8761 | [nan, nan] |
| Cohen's Kappa | 0.8135 | [0.7744, 0.8511] |
| MCC | 0.8135 | [0.7745, 0.8511] |
| AUC-PR (Macro OvR) | 0.9318 | [nan, nan] |

## 4. 每类详细指标（One-vs-Rest）

### Benign_lesions (support=256)

| 指标 | 值 | 95% CI |
|------|-----|--------|
| Precision (PPV) | 0.8588 | — |
| Recall (Sensitivity) | 0.8555 | [0.8117, 0.8967] |
| Specificity | 0.9133 | [0.8834, 0.9399] |
| F1-Score | 0.8571 | [0.8207, 0.8872] |
| NPV | 0.9111 | — |
| AUC-ROC (OvR) | 0.9506 | — |
| AUC-PR (OvR) | 0.9380 | — |

TP=219  FP=36  TN=379  FN=37

### Malignant_lesions (support=210)

| 指标 | 值 | 95% CI |
|------|-----|--------|
| Precision (PPV) | 0.8278 | — |
| Recall (Sensitivity) | 0.8238 | [0.7699, 0.8757] |
| Specificity | 0.9219 | [0.8970, 0.9464] |
| F1-Score | 0.8258 | [0.7867, 0.8641] |
| NPV | 0.9199 | — |
| AUC-ROC (OvR) | 0.9480 | — |
| AUC-PR (OvR) | 0.8716 | — |

TP=173  FP=36  TN=425  FN=37

### body_of_stomach_normal (support=205)

| 指标 | 值 | 95% CI |
|------|-----|--------|
| Precision (PPV) | 0.9469 | — |
| Recall (Sensitivity) | 0.9561 | [0.9268, 0.9835] |
| Specificity | 0.9764 | [0.9610, 0.9891] |
| F1-Score | 0.9515 | [0.9296, 0.9717] |
| NPV | 0.9806 | — |
| AUC-ROC (OvR) | 0.9944 | — |
| AUC-PR (OvR) | 0.9858 | — |

TP=196  FP=11  TN=455  FN=9

## 5. 混淆矩阵

| | Pred: Benign_lesions | Pred: Malignant_lesions | Pred: body_of_stomach_normal |
|---|------|------|------|
| True: Benign_lesions | 219 | 29 | 8 |
| True: Malignant_lesions | 34 | 173 | 3 |
| True: body_of_stomach_normal | 2 | 7 | 196 |

## 6. Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8588    0.8555    0.8571       256
     Malignant_lesions     0.8278    0.8238    0.8258       210
body_of_stomach_normal     0.9469    0.9561    0.9515       205

              accuracy                         0.8763       671
             macro avg     0.8778    0.8785    0.8781       671
          weighted avg     0.8760    0.8763    0.8761       671

```
