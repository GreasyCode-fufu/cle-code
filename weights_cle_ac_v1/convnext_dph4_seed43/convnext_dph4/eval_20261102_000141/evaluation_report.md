# convnext_dph4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph4_seed43/convnext_dph4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9355 |
| Accuracy | 0.9358 |
| Macro-AUROC (OvR) | 0.9937 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9359 |
| Balanced Acc | 0.9355 |
| Macro-Sensitivity | 0.9355 |
| Macro-Specificity | 0.9679 |
| Macro-Precision | 0.9355 |
| AUC-PR (Macro) | 0.9878 |
| Cohen's Kappa | 0.9034 |
| MCC | 0.9035 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 370.9 (2.70 ms/张) |
| End-to-End FPS | 170.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9565 | 0.9231 | 0.9231 | 0.9565 | 0.9912 | 0.9867 | 208 |
| Malignant_lesions | 0.9148 | 0.9575 | 0.9045 | 0.9096 | 0.9623 | 0.9908 | 0.9784 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9896 | 0.9789 | 0.9738 | 0.9845 | 0.9991 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=16 TN=352 FN=16
- **Malignant_lesions**: TP=161 FP=17 TN=383 FN=15
- **body_of_stomach_normal**: TP=186 FP=4 TN=380 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 14 | 2 |
| Malignant_lesions | 13 | 161 | 2 |
| body_of_stomach_normal | 3 | 3 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9231    0.9231    0.9231       208
     Malignant_lesions     0.9045    0.9148    0.9096       176
body_of_stomach_normal     0.9789    0.9688    0.9738       192

              accuracy                         0.9358       576
             macro avg     0.9355    0.9355    0.9355       576
          weighted avg     0.9360    0.9358    0.9359       576

```