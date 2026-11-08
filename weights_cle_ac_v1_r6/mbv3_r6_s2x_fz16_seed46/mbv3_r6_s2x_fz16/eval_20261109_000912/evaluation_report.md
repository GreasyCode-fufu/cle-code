# mbv3_r6_s2x_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2x_fz16_seed46/mbv3_r6_s2x_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9678 |
| Accuracy | 0.9742 |
| Macro-AUROC (OvR) | 0.9976 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9744 |
| Balanced Acc | 0.9712 |
| Macro-Sensitivity | 0.9712 |
| Macro-Specificity | 0.9875 |
| Macro-Precision | 0.9646 |
| AUC-PR (Macro) | 0.9923 |
| Cohen's Kappa | 0.9592 |
| MCC | 0.9592 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.10M |
| FLOPs | 464.20M |
| Params | 3.39M |
| Pure-Inference FPS | 3977.4 (0.25 ms/张) |
| End-to-End FPS | 282.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9641 | 0.9828 | 0.9792 | 0.9716 | 0.9702 | 0.9973 | 0.9969 | 195 |
| Malignant_lesions | 0.9494 | 0.9799 | 0.9146 | 0.9317 | 0.9884 | 0.9955 | 0.9800 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=4 TN=228 FN=7
- **Malignant_lesions**: TP=75 FP=7 TN=341 FN=4
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 7 | 0 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9792    0.9641    0.9716       195
     Malignant_lesions     0.9146    0.9494    0.9317        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9742       427
             macro avg     0.9646    0.9712    0.9678       427
          weighted avg     0.9747    0.9742    0.9744       427

```