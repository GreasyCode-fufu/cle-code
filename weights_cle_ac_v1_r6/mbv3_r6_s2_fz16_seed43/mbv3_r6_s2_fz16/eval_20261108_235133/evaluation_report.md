# mbv3_r6_s2_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_fz16_seed43/mbv3_r6_s2_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9563 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9952 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9565 |
| Balanced Acc | 0.9550 |
| Macro-Sensitivity | 0.9550 |
| Macro-Specificity | 0.9779 |
| Macro-Precision | 0.9586 |
| AUC-PR (Macro) | 0.9912 |
| Cohen's Kappa | 0.9347 |
| MCC | 0.9351 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 4793.4 (0.21 ms/张) |
| End-to-End FPS | 252.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9565 | 0.9266 | 0.9484 | 0.9832 | 0.9924 | 0.9880 | 208 |
| Malignant_lesions | 0.9148 | 0.9875 | 0.9699 | 0.9415 | 0.9634 | 0.9946 | 0.9878 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9896 | 0.9792 | 0.9792 | 0.9896 | 0.9987 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=16 TN=352 FN=6
- **Malignant_lesions**: TP=161 FP=5 TN=395 FN=15
- **body_of_stomach_normal**: TP=188 FP=4 TN=380 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 12 | 161 | 3 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9266    0.9712    0.9484       208
     Malignant_lesions     0.9699    0.9148    0.9415       176
body_of_stomach_normal     0.9792    0.9792    0.9792       192

              accuracy                         0.9566       576
             macro avg     0.9586    0.9550    0.9563       576
          weighted avg     0.9573    0.9566    0.9565       576

```