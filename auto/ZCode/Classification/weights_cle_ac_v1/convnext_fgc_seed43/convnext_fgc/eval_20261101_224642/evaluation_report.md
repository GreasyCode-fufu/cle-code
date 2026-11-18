# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_seed43/convnext_fgc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9617 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9618 |
| Balanced Acc | 0.9617 |
| Macro-Sensitivity | 0.9617 |
| Macro-Specificity | 0.9808 |
| Macro-Precision | 0.9617 |
| AUC-PR (Macro) | 0.9929 |
| Cohen's Kappa | 0.9426 |
| MCC | 0.9426 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 858.4 (1.16 ms/张) |
| End-to-End FPS | 229.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9755 | 0.9565 | 0.9542 | 0.9729 | 0.9950 | 0.9917 | 208 |
| Malignant_lesions | 0.9489 | 0.9800 | 0.9543 | 0.9516 | 0.9776 | 0.9945 | 0.9880 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9870 | 0.9742 | 0.9793 | 0.9921 | 0.9994 | 0.9989 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=9 TN=359 FN=10
- **Malignant_lesions**: TP=167 FP=8 TN=392 FN=9
- **body_of_stomach_normal**: TP=189 FP=5 TN=379 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 6 | 167 | 3 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9565    0.9519    0.9542       208
     Malignant_lesions     0.9543    0.9489    0.9516       176
body_of_stomach_normal     0.9742    0.9844    0.9793       192

              accuracy                         0.9618       576
             macro avg     0.9617    0.9617    0.9617       576
          weighted avg     0.9617    0.9618    0.9618       576

```