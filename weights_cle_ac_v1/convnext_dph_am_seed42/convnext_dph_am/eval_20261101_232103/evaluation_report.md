# convnext_dph_am 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_am_seed42/convnext_dph_am/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9473 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9951 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9478 |
| Balanced Acc | 0.9451 |
| Macro-Sensitivity | 0.9451 |
| Macro-Specificity | 0.9733 |
| Macro-Precision | 0.9520 |
| AUC-PR (Macro) | 0.9912 |
| Cohen's Kappa | 0.9215 |
| MCC | 0.9228 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 29.87M |
| Pure-Inference FPS | 486.3 (2.06 ms/张) |
| End-to-End FPS | 189.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9375 | 0.8987 | 0.9379 | 0.9885 | 0.9940 | 0.9893 | 208 |
| Malignant_lesions | 0.8807 | 0.9850 | 0.9627 | 0.9199 | 0.9494 | 0.9941 | 0.9879 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9974 | 0.9947 | 0.9842 | 0.9871 | 0.9972 | 0.9964 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=23 TN=345 FN=4
- **Malignant_lesions**: TP=155 FP=6 TN=394 FN=21
- **body_of_stomach_normal**: TP=187 FP=1 TN=383 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 4 | 0 |
| Malignant_lesions | 20 | 155 | 1 |
| body_of_stomach_normal | 3 | 2 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8987    0.9808    0.9379       208
     Malignant_lesions     0.9627    0.8807    0.9199       176
body_of_stomach_normal     0.9947    0.9740    0.9842       192

              accuracy                         0.9479       576
             macro avg     0.9520    0.9451    0.9473       576
          weighted avg     0.9503    0.9479    0.9478       576

```