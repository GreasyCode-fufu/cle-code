# convnext_wdd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wdd_seed43/convnext_wdd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9420 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9952 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9426 |
| Balanced Acc | 0.9421 |
| Macro-Sensitivity | 0.9421 |
| Macro-Specificity | 0.9713 |
| Macro-Precision | 0.9420 |
| AUC-PR (Macro) | 0.9908 |
| Cohen's Kappa | 0.9139 |
| MCC | 0.9139 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 29.35M |
| Pure-Inference FPS | 822.1 (1.22 ms/张) |
| End-to-End FPS | 229.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9647 | 0.9372 | 0.9349 | 0.9621 | 0.9937 | 0.9901 | 208 |
| Malignant_lesions | 0.9091 | 0.9650 | 0.9195 | 0.9143 | 0.9602 | 0.9926 | 0.9835 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9844 | 0.9692 | 0.9767 | 0.9921 | 0.9994 | 0.9988 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=13 TN=355 FN=14
- **Malignant_lesions**: TP=160 FP=14 TN=386 FN=16
- **body_of_stomach_normal**: TP=189 FP=6 TN=378 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 13 | 1 |
| Malignant_lesions | 11 | 160 | 5 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9372    0.9327    0.9349       208
     Malignant_lesions     0.9195    0.9091    0.9143       176
body_of_stomach_normal     0.9692    0.9844    0.9767       192

              accuracy                         0.9427       576
             macro avg     0.9420    0.9421    0.9420       576
          weighted avg     0.9425    0.9427    0.9426       576

```