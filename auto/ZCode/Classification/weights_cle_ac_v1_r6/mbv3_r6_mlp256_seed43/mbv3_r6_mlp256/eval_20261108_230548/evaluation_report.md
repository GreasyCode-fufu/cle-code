# mbv3_r6_mlp256 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_mlp256_seed43/mbv3_r6_mlp256/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9527 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9529 |
| Balanced Acc | 0.9514 |
| Macro-Sensitivity | 0.9514 |
| Macro-Specificity | 0.9761 |
| Macro-Precision | 0.9553 |
| AUC-PR (Macro) | 0.9930 |
| Cohen's Kappa | 0.9294 |
| MCC | 0.9300 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.93M |
| FLOPs | 463.85M |
| Params | 3.22M |
| Pure-Inference FPS | 4207.6 (0.24 ms/张) |
| End-to-End FPS | 226.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9565 | 0.9263 | 0.9459 | 0.9805 | 0.9940 | 0.9902 | 208 |
| Malignant_lesions | 0.9034 | 0.9900 | 0.9755 | 0.9381 | 0.9588 | 0.9959 | 0.9907 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9818 | 0.9643 | 0.9742 | 0.9921 | 0.9991 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=16 TN=352 FN=7
- **Malignant_lesions**: TP=159 FP=4 TN=396 FN=17
- **body_of_stomach_normal**: TP=189 FP=7 TN=377 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 4 | 3 |
| Malignant_lesions | 13 | 159 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9263    0.9663    0.9459       208
     Malignant_lesions     0.9755    0.9034    0.9381       176
body_of_stomach_normal     0.9643    0.9844    0.9742       192

              accuracy                         0.9531       576
             macro avg     0.9553    0.9514    0.9527       576
          weighted avg     0.9540    0.9531    0.9529       576

```