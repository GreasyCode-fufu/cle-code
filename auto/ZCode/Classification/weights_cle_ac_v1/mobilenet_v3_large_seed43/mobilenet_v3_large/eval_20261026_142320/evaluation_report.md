# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/mobilenet_v3_large_seed43/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9388 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9912 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9389 |
| Balanced Acc | 0.9378 |
| Macro-Sensitivity | 0.9378 |
| Macro-Specificity | 0.9691 |
| Macro-Precision | 0.9410 |
| AUC-PR (Macro) | 0.9829 |
| Cohen's Kappa | 0.9085 |
| MCC | 0.9090 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 5362.8 (0.19 ms/张) |
| End-to-End FPS | 299.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9484 | 0.9112 | 0.9242 | 0.9641 | 0.9861 | 0.9774 | 208 |
| Malignant_lesions | 0.8864 | 0.9825 | 0.9571 | 0.9204 | 0.9516 | 0.9883 | 0.9729 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9766 | 0.9548 | 0.9719 | 0.9947 | 0.9992 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=19 TN=349 FN=13
- **Malignant_lesions**: TP=156 FP=7 TN=393 FN=20
- **body_of_stomach_normal**: TP=190 FP=9 TN=375 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 7 | 6 |
| Malignant_lesions | 17 | 156 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9112    0.9375    0.9242       208
     Malignant_lesions     0.9571    0.8864    0.9204       176
body_of_stomach_normal     0.9548    0.9896    0.9719       192

              accuracy                         0.9392       576
             macro avg     0.9410    0.9378    0.9388       576
          weighted avg     0.9397    0.9392    0.9389       576

```