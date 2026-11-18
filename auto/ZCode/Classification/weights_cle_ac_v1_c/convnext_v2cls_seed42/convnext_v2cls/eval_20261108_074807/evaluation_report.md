# convnext_v2cls 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2cls_seed42/convnext_v2cls/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9506 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9885 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9510 |
| Balanced Acc | 0.9497 |
| Macro-Sensitivity | 0.9497 |
| Macro-Specificity | 0.9754 |
| Macro-Precision | 0.9525 |
| AUC-PR (Macro) | 0.9766 |
| Cohen's Kappa | 0.9268 |
| MCC | 0.9273 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 480.8 (2.08 ms/张) |
| End-to-End FPS | 187.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9620 | 0.9343 | 0.9454 | 0.9752 | 0.9865 | 0.9556 | 208 |
| Malignant_lesions | 0.8977 | 0.9850 | 0.9634 | 0.9294 | 0.9563 | 0.9794 | 0.9752 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9792 | 0.9598 | 0.9770 | 0.9973 | 0.9995 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=14 TN=354 FN=9
- **Malignant_lesions**: TP=158 FP=6 TN=394 FN=18
- **body_of_stomach_normal**: TP=191 FP=8 TN=376 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 6 | 3 |
| Malignant_lesions | 13 | 158 | 5 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9343    0.9567    0.9454       208
     Malignant_lesions     0.9634    0.8977    0.9294       176
body_of_stomach_normal     0.9598    0.9948    0.9770       192

              accuracy                         0.9514       576
             macro avg     0.9525    0.9497    0.9506       576
          weighted avg     0.9517    0.9514    0.9510       576

```