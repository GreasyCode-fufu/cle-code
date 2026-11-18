# convnext_v4c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4c_seed43/convnext_v4c/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9725 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9979 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9766 |
| Balanced Acc | 0.9754 |
| Macro-Sensitivity | 0.9754 |
| Macro-Specificity | 0.9885 |
| Macro-Precision | 0.9699 |
| AUC-PR (Macro) | 0.9936 |
| Cohen's Kappa | 0.9629 |
| MCC | 0.9630 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.85G |
| FLOPs | 3.71G |
| Params | 3.45M |
| Pure-Inference FPS | 759.6 (1.32 ms/张) |
| End-to-End FPS | 222.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9641 | 0.9871 | 0.9843 | 0.9741 | 0.9703 | 0.9973 | 0.9970 | 195 |
| Malignant_lesions | 0.9620 | 0.9856 | 0.9383 | 0.9500 | 0.9913 | 0.9964 | 0.9838 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 1.0000 | 0.9999 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=3 TN=229 FN=7
- **Malignant_lesions**: TP=76 FP=5 TN=343 FN=3
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 5 | 2 |
| Malignant_lesions | 3 | 76 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9843    0.9641    0.9741       195
     Malignant_lesions     0.9383    0.9620    0.9500        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9766       427
             macro avg     0.9699    0.9754    0.9725       427
          weighted avg     0.9768    0.9766    0.9766       427

```