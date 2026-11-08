# convnext_v5pf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5pf_seed43/convnext_v5pf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9618 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9618 |
| Balanced Acc | 0.9623 |
| Macro-Sensitivity | 0.9623 |
| Macro-Specificity | 0.9810 |
| Macro-Precision | 0.9613 |
| AUC-PR (Macro) | 0.9926 |
| Cohen's Kappa | 0.9426 |
| MCC | 0.9427 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1284.5 (0.78 ms/张) |
| End-to-End FPS | 259.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9783 | 0.9608 | 0.9515 | 0.9677 | 0.9942 | 0.9907 | 208 |
| Malignant_lesions | 0.9602 | 0.9725 | 0.9389 | 0.9494 | 0.9823 | 0.9949 | 0.9886 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9922 | 0.9844 | 0.9844 | 0.9922 | 0.9993 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=8 TN=360 FN=12
- **Malignant_lesions**: TP=169 FP=11 TN=389 FN=7
- **body_of_stomach_normal**: TP=189 FP=3 TN=381 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 11 | 1 |
| Malignant_lesions | 5 | 169 | 2 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9608    0.9423    0.9515       208
     Malignant_lesions     0.9389    0.9602    0.9494       176
body_of_stomach_normal     0.9844    0.9844    0.9844       192

              accuracy                         0.9618       576
             macro avg     0.9613    0.9623    0.9618       576
          weighted avg     0.9620    0.9618    0.9618       576

```