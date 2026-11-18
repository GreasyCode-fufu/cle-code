# convnext_v5grn4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5grn4_seed42/convnext_v5grn4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9617 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9974 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9619 |
| Balanced Acc | 0.9632 |
| Macro-Sensitivity | 0.9632 |
| Macro-Specificity | 0.9813 |
| Macro-Precision | 0.9614 |
| AUC-PR (Macro) | 0.9952 |
| Cohen's Kappa | 0.9427 |
| MCC | 0.9432 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 583.3 (1.71 ms/张) |
| End-to-End FPS | 198.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9891 | 0.9797 | 0.9531 | 0.9604 | 0.9960 | 0.9938 | 208 |
| Malignant_lesions | 0.9773 | 0.9600 | 0.9149 | 0.9451 | 0.9897 | 0.9970 | 0.9934 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9948 | 0.9895 | 0.9869 | 0.9922 | 0.9993 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=4 TN=364 FN=15
- **Malignant_lesions**: TP=172 FP=16 TN=384 FN=4
- **body_of_stomach_normal**: TP=189 FP=2 TN=382 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 14 | 1 |
| Malignant_lesions | 3 | 172 | 1 |
| body_of_stomach_normal | 1 | 2 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9797    0.9279    0.9531       208
     Malignant_lesions     0.9149    0.9773    0.9451       176
body_of_stomach_normal     0.9895    0.9844    0.9869       192

              accuracy                         0.9618       576
             macro avg     0.9614    0.9632    0.9617       576
          weighted avg     0.9632    0.9618    0.9619       576

```