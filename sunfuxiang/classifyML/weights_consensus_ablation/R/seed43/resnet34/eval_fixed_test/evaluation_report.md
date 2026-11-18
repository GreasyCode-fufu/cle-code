# resnet34 评估报告

- 权重: `/home/login/data/cle/sunfuxiang/classifyML/weights_consensus_ablation/R/seed43/resnet34/resnet34_R_seed43_best.pth`
- 测试集: `/home/login/data/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9035 |
| Accuracy | 0.9016 |
| Macro-AUROC (OvR) | 0.9791 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9008 |
| Balanced Acc | 0.9074 |
| Macro-Sensitivity | 0.9074 |
| Macro-Specificity | 0.9508 |
| Macro-Precision | 0.9017 |
| AUC-PR (Macro) | 0.9622 |
| Cohen's Kappa | 0.8522 |
| MCC | 0.8534 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3794.5 (0.26 ms/张) |
| End-to-End FPS | 241.5 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8320 | 0.9518 | 0.9142 | 0.8712 | 0.9018 | 0.9688 | 0.9592 | 256 |
| Malignant_lesions | 0.9048 | 0.9306 | 0.8559 | 0.8796 | 0.9555 | 0.9716 | 0.9345 | 210 |
| body_of_stomach_normal | 0.9854 | 0.9700 | 0.9352 | 0.9596 | 0.9934 | 0.9969 | 0.9928 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=213 FP=20 TN=395 FN=43
- **Malignant_lesions**: TP=190 FP=32 TN=429 FN=20
- **body_of_stomach_normal**: TP=202 FP=14 TN=452 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 213 | 31 | 12 |
| Malignant_lesions | 18 | 190 | 2 |
| body_of_stomach_normal | 2 | 1 | 202 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9142    0.8320    0.8712       256
     Malignant_lesions     0.8559    0.9048    0.8796       210
body_of_stomach_normal     0.9352    0.9854    0.9596       205

              accuracy                         0.9016       671
             macro avg     0.9017    0.9074    0.9035       671
          weighted avg     0.9023    0.9016    0.9008       671

```