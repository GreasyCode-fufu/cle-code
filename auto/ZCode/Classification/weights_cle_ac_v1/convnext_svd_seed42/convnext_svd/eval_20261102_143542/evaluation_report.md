# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_svd_seed42/convnext_svd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9561 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9932 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9564 |
| Balanced Acc | 0.9550 |
| Macro-Sensitivity | 0.9550 |
| Macro-Specificity | 0.9779 |
| Macro-Precision | 0.9579 |
| AUC-PR (Macro) | 0.9868 |
| Cohen's Kappa | 0.9347 |
| MCC | 0.9350 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 733.0 (1.36 ms/张) |
| End-to-End FPS | 209.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9592 | 0.9306 | 0.9481 | 0.9806 | 0.9917 | 0.9822 | 208 |
| Malignant_lesions | 0.9091 | 0.9850 | 0.9639 | 0.9357 | 0.9610 | 0.9882 | 0.9789 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9896 | 0.9794 | 0.9845 | 0.9948 | 0.9996 | 0.9993 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=15 TN=353 FN=7
- **Malignant_lesions**: TP=160 FP=6 TN=394 FN=16
- **body_of_stomach_normal**: TP=190 FP=4 TN=380 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 13 | 160 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9306    0.9663    0.9481       208
     Malignant_lesions     0.9639    0.9091    0.9357       176
body_of_stomach_normal     0.9794    0.9896    0.9845       192

              accuracy                         0.9566       576
             macro avg     0.9579    0.9550    0.9561       576
          weighted avg     0.9570    0.9566    0.9564       576

```