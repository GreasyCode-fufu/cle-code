# convnext_tiny_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_seed42/convnext_tiny_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9426 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9921 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9428 |
| Balanced Acc | 0.9422 |
| Macro-Sensitivity | 0.9422 |
| Macro-Specificity | 0.9712 |
| Macro-Precision | 0.9429 |
| AUC-PR (Macro) | 0.9853 |
| Cohen's Kappa | 0.9138 |
| MCC | 0.9139 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 994.9 (1.01 ms/张) |
| End-to-End FPS | 45.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9565 | 0.9242 | 0.9308 | 0.9644 | 0.9883 | 0.9824 | 208 |
| Malignant_lesions | 0.9205 | 0.9675 | 0.9257 | 0.9231 | 0.9651 | 0.9902 | 0.9775 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9896 | 0.9789 | 0.9738 | 0.9845 | 0.9978 | 0.9959 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=16 TN=352 FN=13
- **Malignant_lesions**: TP=162 FP=13 TN=387 FN=14
- **body_of_stomach_normal**: TP=186 FP=4 TN=380 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 12 | 1 |
| Malignant_lesions | 11 | 162 | 3 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9242    0.9375    0.9308       208
     Malignant_lesions     0.9257    0.9205    0.9231       176
body_of_stomach_normal     0.9789    0.9688    0.9738       192

              accuracy                         0.9427       576
             macro avg     0.9429    0.9422    0.9426       576
          weighted avg     0.9429    0.9427    0.9428       576

```