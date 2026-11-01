# convnext_dpea2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dpea2_seed44/convnext_dpea2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9419 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9940 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9424 |
| Balanced Acc | 0.9423 |
| Macro-Sensitivity | 0.9423 |
| Macro-Specificity | 0.9714 |
| Macro-Precision | 0.9424 |
| AUC-PR (Macro) | 0.9888 |
| Cohen's Kappa | 0.9139 |
| MCC | 0.9143 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 28.12M |
| Pure-Inference FPS | 640.1 (1.56 ms/张) |
| End-to-End FPS | 203.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9755 | 0.9552 | 0.9389 | 0.9573 | 0.9911 | 0.9878 | 208 |
| Malignant_lesions | 0.9091 | 0.9725 | 0.9357 | 0.9222 | 0.9605 | 0.9920 | 0.9808 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9661 | 0.9363 | 0.9646 | 0.9973 | 0.9989 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=9 TN=359 FN=16
- **Malignant_lesions**: TP=160 FP=11 TN=389 FN=16
- **body_of_stomach_normal**: TP=191 FP=13 TN=371 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 11 | 5 |
| Malignant_lesions | 8 | 160 | 8 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9552    0.9231    0.9389       208
     Malignant_lesions     0.9357    0.9091    0.9222       176
body_of_stomach_normal     0.9363    0.9948    0.9646       192

              accuracy                         0.9427       576
             macro avg     0.9424    0.9423    0.9419       576
          weighted avg     0.9429    0.9427    0.9424       576

```