# convnext_v6fx 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v6fx_seed43/convnext_v6fx/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9317 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9924 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9322 |
| Balanced Acc | 0.9284 |
| Macro-Sensitivity | 0.9284 |
| Macro-Specificity | 0.9651 |
| Macro-Precision | 0.9407 |
| AUC-PR (Macro) | 0.9855 |
| Cohen's Kappa | 0.8978 |
| MCC | 0.9007 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.70M |
| Pure-Inference FPS | 660.7 (1.51 ms/张) |
| End-to-End FPS | 210.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9103 | 0.8602 | 0.9144 | 0.9853 | 0.9909 | 0.9856 | 208 |
| Malignant_lesions | 0.8352 | 0.9875 | 0.9671 | 0.8963 | 0.9316 | 0.9873 | 0.9726 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9974 | 0.9947 | 0.9842 | 0.9871 | 0.9989 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=33 TN=335 FN=5
- **Malignant_lesions**: TP=147 FP=5 TN=395 FN=29
- **body_of_stomach_normal**: TP=187 FP=1 TN=383 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 5 | 0 |
| Malignant_lesions | 28 | 147 | 1 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8602    0.9760    0.9144       208
     Malignant_lesions     0.9671    0.8352    0.8963       176
body_of_stomach_normal     0.9947    0.9740    0.9842       192

              accuracy                         0.9323       576
             macro avg     0.9407    0.9284    0.9317       576
          weighted avg     0.9377    0.9323    0.9322       576

```