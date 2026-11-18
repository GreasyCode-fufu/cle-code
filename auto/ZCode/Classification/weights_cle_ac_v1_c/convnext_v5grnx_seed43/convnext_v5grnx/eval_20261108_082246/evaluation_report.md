# convnext_v5grnx 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5grnx_seed43/convnext_v5grnx/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9614 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9962 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9617 |
| Balanced Acc | 0.9621 |
| Macro-Sensitivity | 0.9621 |
| Macro-Specificity | 0.9811 |
| Macro-Precision | 0.9610 |
| AUC-PR (Macro) | 0.9930 |
| Cohen's Kappa | 0.9426 |
| MCC | 0.9427 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.69M |
| Pure-Inference FPS | 677.7 (1.48 ms/张) |
| End-to-End FPS | 203.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9810 | 0.9653 | 0.9512 | 0.9652 | 0.9938 | 0.9913 | 208 |
| Malignant_lesions | 0.9489 | 0.9700 | 0.9330 | 0.9408 | 0.9773 | 0.9951 | 0.9884 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9922 | 0.9846 | 0.9922 | 1.0000 | 0.9996 | 0.9993 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=7 TN=361 FN=13
- **Malignant_lesions**: TP=167 FP=12 TN=388 FN=9
- **body_of_stomach_normal**: TP=192 FP=3 TN=381 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 12 | 1 |
| Malignant_lesions | 7 | 167 | 2 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9653    0.9375    0.9512       208
     Malignant_lesions     0.9330    0.9489    0.9408       176
body_of_stomach_normal     0.9846    1.0000    0.9922       192

              accuracy                         0.9618       576
             macro avg     0.9610    0.9621    0.9614       576
          weighted avg     0.9619    0.9618    0.9617       576

```