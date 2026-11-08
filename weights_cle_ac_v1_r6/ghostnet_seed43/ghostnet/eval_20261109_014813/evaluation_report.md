# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/ghostnet_seed43/ghostnet/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9400 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9885 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9406 |
| Balanced Acc | 0.9378 |
| Macro-Sensitivity | 0.9378 |
| Macro-Specificity | 0.9698 |
| Macro-Precision | 0.9453 |
| AUC-PR (Macro) | 0.9780 |
| Cohen's Kappa | 0.9110 |
| MCC | 0.9125 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 2794.7 (0.36 ms/张) |
| End-to-End FPS | 28.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9321 | 0.8899 | 0.9287 | 0.9828 | 0.9837 | 0.9721 | 208 |
| Malignant_lesions | 0.8580 | 0.9850 | 0.9618 | 0.9069 | 0.9403 | 0.9825 | 0.9631 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9922 | 0.9844 | 0.9844 | 0.9922 | 0.9993 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=25 TN=343 FN=6
- **Malignant_lesions**: TP=151 FP=6 TN=394 FN=25
- **body_of_stomach_normal**: TP=189 FP=3 TN=381 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 23 | 151 | 2 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8899    0.9712    0.9287       208
     Malignant_lesions     0.9618    0.8580    0.9069       176
body_of_stomach_normal     0.9844    0.9844    0.9844       192

              accuracy                         0.9410       576
             macro avg     0.9453    0.9378    0.9400       576
          weighted avg     0.9433    0.9410    0.9406       576

```