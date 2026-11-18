# convnext_v5grn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5grn_seed42/convnext_v5grn/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9476 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9953 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9476 |
| Balanced Acc | 0.9495 |
| Macro-Sensitivity | 0.9495 |
| Macro-Specificity | 0.9743 |
| Macro-Precision | 0.9474 |
| AUC-PR (Macro) | 0.9908 |
| Cohen's Kappa | 0.9218 |
| MCC | 0.9226 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 364.8 (2.74 ms/张) |
| End-to-End FPS | 167.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8990 | 0.9864 | 0.9740 | 0.9350 | 0.9453 | 0.9942 | 0.9896 | 208 |
| Malignant_lesions | 0.9545 | 0.9625 | 0.9180 | 0.9359 | 0.9796 | 0.9926 | 0.9845 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9740 | 0.9502 | 0.9720 | 0.9973 | 0.9992 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=5 TN=363 FN=21
- **Malignant_lesions**: TP=168 FP=15 TN=385 FN=8
- **body_of_stomach_normal**: TP=191 FP=10 TN=374 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 15 | 6 |
| Malignant_lesions | 4 | 168 | 4 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9740    0.8990    0.9350       208
     Malignant_lesions     0.9180    0.9545    0.9359       176
body_of_stomach_normal     0.9502    0.9948    0.9720       192

              accuracy                         0.9479       576
             macro avg     0.9474    0.9495    0.9476       576
          weighted avg     0.9490    0.9479    0.9476       576

```