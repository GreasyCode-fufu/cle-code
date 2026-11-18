# convnext_cmp_fse 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_cmp_fse_seed42/convnext_cmp_fse/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9400 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9959 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9405 |
| Balanced Acc | 0.9381 |
| Macro-Sensitivity | 0.9381 |
| Macro-Specificity | 0.9698 |
| Macro-Precision | 0.9446 |
| AUC-PR (Macro) | 0.9923 |
| Cohen's Kappa | 0.9110 |
| MCC | 0.9123 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 500.8 (2.00 ms/张) |
| End-to-End FPS | 186.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9402 | 0.9013 | 0.9327 | 0.9802 | 0.9943 | 0.9908 | 208 |
| Malignant_lesions | 0.8636 | 0.9875 | 0.9682 | 0.9129 | 0.9427 | 0.9951 | 0.9889 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9818 | 0.9643 | 0.9742 | 0.9921 | 0.9983 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=22 TN=346 FN=7
- **Malignant_lesions**: TP=152 FP=5 TN=395 FN=24
- **body_of_stomach_normal**: TP=189 FP=7 TN=377 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 19 | 152 | 5 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9013    0.9663    0.9327       208
     Malignant_lesions     0.9682    0.8636    0.9129       176
body_of_stomach_normal     0.9643    0.9844    0.9742       192

              accuracy                         0.9410       576
             macro avg     0.9446    0.9381    0.9400       576
          weighted avg     0.9427    0.9410    0.9405       576

```