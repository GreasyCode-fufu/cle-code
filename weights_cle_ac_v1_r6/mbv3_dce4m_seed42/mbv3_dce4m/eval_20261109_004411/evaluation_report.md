# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce4m_seed42/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9551 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9549 |
| Balanced Acc | 0.9552 |
| Macro-Sensitivity | 0.9552 |
| Macro-Specificity | 0.9773 |
| Macro-Precision | 0.9551 |
| AUC-PR (Macro) | 0.9904 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9322 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5637.4 (0.18 ms/张) |
| End-to-End FPS | 239.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9647 | 0.9375 | 0.9375 | 0.9647 | 0.9920 | 0.9861 | 208 |
| Malignant_lesions | 0.9489 | 0.9725 | 0.9382 | 0.9435 | 0.9774 | 0.9936 | 0.9862 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9948 | 0.9895 | 0.9843 | 0.9896 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=13 TN=355 FN=13
- **Malignant_lesions**: TP=167 FP=11 TN=389 FN=9
- **body_of_stomach_normal**: TP=188 FP=2 TN=382 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 11 | 2 |
| Malignant_lesions | 9 | 167 | 0 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9375    0.9375    0.9375       208
     Malignant_lesions     0.9382    0.9489    0.9435       176
body_of_stomach_normal     0.9895    0.9792    0.9843       192

              accuracy                         0.9549       576
             macro avg     0.9551    0.9552    0.9551       576
          weighted avg     0.9550    0.9549    0.9549       576

```