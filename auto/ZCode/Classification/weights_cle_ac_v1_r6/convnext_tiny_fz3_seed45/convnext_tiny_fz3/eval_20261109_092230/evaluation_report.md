# convnext_tiny_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_seed45/convnext_tiny_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9458 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9932 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9461 |
| Balanced Acc | 0.9450 |
| Macro-Sensitivity | 0.9450 |
| Macro-Specificity | 0.9728 |
| Macro-Precision | 0.9470 |
| AUC-PR (Macro) | 0.9872 |
| Cohen's Kappa | 0.9190 |
| MCC | 0.9192 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 960.9 (1.04 ms/张) |
| End-to-End FPS | 97.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9565 | 0.9252 | 0.9384 | 0.9724 | 0.9920 | 0.9874 | 208 |
| Malignant_lesions | 0.9091 | 0.9775 | 0.9467 | 0.9275 | 0.9607 | 0.9902 | 0.9791 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9844 | 0.9689 | 0.9714 | 0.9869 | 0.9976 | 0.9950 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=16 TN=352 FN=10
- **Malignant_lesions**: TP=160 FP=9 TN=391 FN=16
- **body_of_stomach_normal**: TP=187 FP=6 TN=378 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 8 | 2 |
| Malignant_lesions | 12 | 160 | 4 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9252    0.9519    0.9384       208
     Malignant_lesions     0.9467    0.9091    0.9275       176
body_of_stomach_normal     0.9689    0.9740    0.9714       192

              accuracy                         0.9462       576
             macro avg     0.9470    0.9450    0.9458       576
          weighted avg     0.9464    0.9462    0.9461       576

```