# convnext_v4f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4f_seed42/convnext_v4f/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9619 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9964 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9619 |
| Balanced Acc | 0.9607 |
| Macro-Sensitivity | 0.9607 |
| Macro-Specificity | 0.9805 |
| Macro-Precision | 0.9636 |
| AUC-PR (Macro) | 0.9929 |
| Cohen's Kappa | 0.9425 |
| MCC | 0.9428 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 1436.4 (0.70 ms/张) |
| End-to-End FPS | 263.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9592 | 0.9309 | 0.9506 | 0.9833 | 0.9946 | 0.9919 | 208 |
| Malignant_lesions | 0.9318 | 0.9875 | 0.9704 | 0.9507 | 0.9705 | 0.9950 | 0.9877 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9948 | 0.9895 | 0.9843 | 0.9896 | 0.9996 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=15 TN=353 FN=6
- **Malignant_lesions**: TP=164 FP=5 TN=395 FN=12
- **body_of_stomach_normal**: TP=188 FP=2 TN=382 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 11 | 164 | 1 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9309    0.9712    0.9506       208
     Malignant_lesions     0.9704    0.9318    0.9507       176
body_of_stomach_normal     0.9895    0.9792    0.9843       192

              accuracy                         0.9618       576
             macro avg     0.9636    0.9607    0.9619       576
          weighted avg     0.9625    0.9618    0.9619       576

```