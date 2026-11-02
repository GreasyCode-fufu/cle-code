# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_ef_seed43/convnext_ef/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9598 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9957 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9601 |
| Balanced Acc | 0.9598 |
| Macro-Sensitivity | 0.9598 |
| Macro-Specificity | 0.9800 |
| Macro-Precision | 0.9598 |
| AUC-PR (Macro) | 0.9920 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9400 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 936.7 (1.07 ms/张) |
| End-to-End FPS | 233.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9701 | 0.9471 | 0.9471 | 0.9701 | 0.9933 | 0.9900 | 208 |
| Malignant_lesions | 0.9375 | 0.9725 | 0.9375 | 0.9375 | 0.9725 | 0.9939 | 0.9862 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9974 | 0.9948 | 0.9948 | 0.9974 | 0.9998 | 0.9997 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=11 TN=357 FN=11
- **Malignant_lesions**: TP=165 FP=11 TN=389 FN=11
- **body_of_stomach_normal**: TP=191 FP=1 TN=383 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 11 | 165 | 0 |
| body_of_stomach_normal | 0 | 1 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9471    0.9471    0.9471       208
     Malignant_lesions     0.9375    0.9375    0.9375       176
body_of_stomach_normal     0.9948    0.9948    0.9948       192

              accuracy                         0.9601       576
             macro avg     0.9598    0.9598    0.9598       576
          weighted avg     0.9601    0.9601    0.9601       576

```