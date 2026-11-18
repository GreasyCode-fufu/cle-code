# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2c_seed42/convnext_v2c/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9594 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9598 |
| Balanced Acc | 0.9586 |
| Macro-Sensitivity | 0.9586 |
| Macro-Specificity | 0.9798 |
| Macro-Precision | 0.9609 |
| AUC-PR (Macro) | 0.9936 |
| Cohen's Kappa | 0.9399 |
| MCC | 0.9402 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 841.6 (1.19 ms/张) |
| End-to-End FPS | 223.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9701 | 0.9481 | 0.9571 | 0.9808 | 0.9940 | 0.9920 | 208 |
| Malignant_lesions | 0.9148 | 0.9875 | 0.9699 | 0.9415 | 0.9634 | 0.9962 | 0.9911 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9818 | 0.9646 | 0.9795 | 0.9974 | 0.9989 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=11 TN=357 FN=7
- **Malignant_lesions**: TP=161 FP=5 TN=395 FN=15
- **body_of_stomach_normal**: TP=191 FP=7 TN=377 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 10 | 161 | 5 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9481    0.9663    0.9571       208
     Malignant_lesions     0.9699    0.9148    0.9415       176
body_of_stomach_normal     0.9646    0.9948    0.9795       192

              accuracy                         0.9601       576
             macro avg     0.9609    0.9586    0.9594       576
          weighted avg     0.9603    0.9601    0.9598       576

```