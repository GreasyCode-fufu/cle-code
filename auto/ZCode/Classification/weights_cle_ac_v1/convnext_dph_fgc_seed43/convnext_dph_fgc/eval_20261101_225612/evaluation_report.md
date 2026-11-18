# convnext_dph_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_fgc_seed43/convnext_dph_fgc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9453 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9943 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9459 |
| Balanced Acc | 0.9458 |
| Macro-Sensitivity | 0.9458 |
| Macro-Specificity | 0.9732 |
| Macro-Precision | 0.9452 |
| AUC-PR (Macro) | 0.9887 |
| Cohen's Kappa | 0.9191 |
| MCC | 0.9193 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 28.59M |
| Pure-Inference FPS | 582.1 (1.72 ms/张) |
| End-to-End FPS | 203.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9755 | 0.9554 | 0.9415 | 0.9599 | 0.9933 | 0.9893 | 208 |
| Malignant_lesions | 0.9148 | 0.9675 | 0.9253 | 0.9200 | 0.9627 | 0.9904 | 0.9781 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9766 | 0.9550 | 0.9745 | 0.9973 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=9 TN=359 FN=15
- **Malignant_lesions**: TP=161 FP=13 TN=387 FN=15
- **body_of_stomach_normal**: TP=191 FP=9 TN=375 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 13 | 2 |
| Malignant_lesions | 8 | 161 | 7 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9554    0.9279    0.9415       208
     Malignant_lesions     0.9253    0.9148    0.9200       176
body_of_stomach_normal     0.9550    0.9948    0.9745       192

              accuracy                         0.9462       576
             macro avg     0.9452    0.9458    0.9453       576
          weighted avg     0.9461    0.9462    0.9459       576

```