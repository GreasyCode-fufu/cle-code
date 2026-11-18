# convnext_v5xrm2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5xrm2_seed46/convnext_v5xrm2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9543 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9966 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9547 |
| Balanced Acc | 0.9541 |
| Macro-Sensitivity | 0.9541 |
| Macro-Specificity | 0.9773 |
| Macro-Precision | 0.9546 |
| AUC-PR (Macro) | 0.9933 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9322 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.69M |
| Pure-Inference FPS | 899.8 (1.11 ms/张) |
| End-to-End FPS | 232.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9674 | 0.9426 | 0.9448 | 0.9700 | 0.9951 | 0.9921 | 208 |
| Malignant_lesions | 0.9205 | 0.9750 | 0.9419 | 0.9310 | 0.9653 | 0.9953 | 0.9887 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9896 | 0.9795 | 0.9871 | 0.9974 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=12 TN=356 FN=11
- **Malignant_lesions**: TP=162 FP=10 TN=390 FN=14
- **body_of_stomach_normal**: TP=191 FP=4 TN=380 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 11 | 162 | 3 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9426    0.9471    0.9448       208
     Malignant_lesions     0.9419    0.9205    0.9310       176
body_of_stomach_normal     0.9795    0.9948    0.9871       192

              accuracy                         0.9549       576
             macro avg     0.9546    0.9541    0.9543       576
          weighted avg     0.9547    0.9549    0.9547       576

```