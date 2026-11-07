# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4e_seed43/convnext_v4e/model_best.pth`
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
| Balanced Acc | 0.9603 |
| Macro-Sensitivity | 0.9603 |
| Macro-Specificity | 0.9806 |
| Macro-Precision | 0.9633 |
| AUC-PR (Macro) | 0.9928 |
| Cohen's Kappa | 0.9425 |
| MCC | 0.9429 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 1561.3 (0.64 ms/张) |
| End-to-End FPS | 270.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9647 | 0.9398 | 0.9575 | 0.9861 | 0.9946 | 0.9909 | 208 |
| Malignant_lesions | 0.9205 | 0.9900 | 0.9759 | 0.9474 | 0.9659 | 0.9949 | 0.9894 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9870 | 0.9742 | 0.9793 | 0.9921 | 0.9990 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=13 TN=355 FN=5
- **Malignant_lesions**: TP=162 FP=4 TN=396 FN=14
- **body_of_stomach_normal**: TP=189 FP=5 TN=379 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 4 | 1 |
| Malignant_lesions | 10 | 162 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9398    0.9760    0.9575       208
     Malignant_lesions     0.9759    0.9205    0.9474       176
body_of_stomach_normal     0.9742    0.9844    0.9793       192

              accuracy                         0.9618       576
             macro avg     0.9633    0.9603    0.9614       576
          weighted avg     0.9623    0.9618    0.9617       576

```