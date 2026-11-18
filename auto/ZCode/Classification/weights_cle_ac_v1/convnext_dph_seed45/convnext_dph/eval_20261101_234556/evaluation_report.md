# convnext_dph 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_seed45/convnext_dph/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9631 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9959 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9634 |
| Balanced Acc | 0.9630 |
| Macro-Sensitivity | 0.9630 |
| Macro-Specificity | 0.9817 |
| Macro-Precision | 0.9633 |
| AUC-PR (Macro) | 0.9922 |
| Cohen's Kappa | 0.9452 |
| MCC | 0.9452 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 1702.9 (0.59 ms/张) |
| End-to-End FPS | 270.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9755 | 0.9567 | 0.9567 | 0.9755 | 0.9942 | 0.9890 | 208 |
| Malignant_lesions | 0.9375 | 0.9800 | 0.9538 | 0.9456 | 0.9727 | 0.9937 | 0.9880 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9896 | 0.9795 | 0.9871 | 0.9974 | 0.9997 | 0.9995 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=9 TN=359 FN=9
- **Malignant_lesions**: TP=165 FP=8 TN=392 FN=11
- **body_of_stomach_normal**: TP=191 FP=4 TN=380 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 9 | 165 | 2 |
| body_of_stomach_normal | 0 | 1 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9567    0.9567    0.9567       208
     Malignant_lesions     0.9538    0.9375    0.9456       176
body_of_stomach_normal     0.9795    0.9948    0.9871       192

              accuracy                         0.9635       576
             macro avg     0.9633    0.9630    0.9631       576
          weighted avg     0.9634    0.9635    0.9634       576

```