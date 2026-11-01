# convnext_dph_lsdg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_lsdg_seed42/convnext_dph_lsdg/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9542 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9944 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9546 |
| Balanced Acc | 0.9535 |
| Macro-Sensitivity | 0.9535 |
| Macro-Specificity | 0.9772 |
| Macro-Precision | 0.9556 |
| AUC-PR (Macro) | 0.9887 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9324 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.27G |
| Params | 28.93M |
| Pure-Inference FPS | 547.3 (1.83 ms/张) |
| End-to-End FPS | 197.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9674 | 0.9431 | 0.9499 | 0.9753 | 0.9928 | 0.9879 | 208 |
| Malignant_lesions | 0.9091 | 0.9850 | 0.9639 | 0.9357 | 0.9610 | 0.9911 | 0.9795 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9792 | 0.9598 | 0.9770 | 0.9973 | 0.9993 | 0.9988 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=12 TN=356 FN=9
- **Malignant_lesions**: TP=160 FP=6 TN=394 FN=16
- **body_of_stomach_normal**: TP=191 FP=8 TN=376 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 6 | 3 |
| Malignant_lesions | 11 | 160 | 5 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9431    0.9567    0.9499       208
     Malignant_lesions     0.9639    0.9091    0.9357       176
body_of_stomach_normal     0.9598    0.9948    0.9770       192

              accuracy                         0.9549       576
             macro avg     0.9556    0.9535    0.9542       576
          weighted avg     0.9550    0.9549    0.9546       576

```