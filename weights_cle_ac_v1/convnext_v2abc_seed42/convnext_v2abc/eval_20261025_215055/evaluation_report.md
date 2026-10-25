# convnext_v2abc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2abc_seed42/convnext_v2abc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9453 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9936 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9459 |
| Balanced Acc | 0.9437 |
| Macro-Sensitivity | 0.9437 |
| Macro-Specificity | 0.9725 |
| Macro-Precision | 0.9489 |
| AUC-PR (Macro) | 0.9887 |
| Cohen's Kappa | 0.9189 |
| MCC | 0.9198 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.66G |
| FLOPs | 9.31G |
| Params | 28.67M |
| Pure-Inference FPS | 538.6 (1.86 ms/张) |
| End-to-End FPS | 194.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9457 | 0.9099 | 0.9395 | 0.9831 | 0.9924 | 0.9864 | 208 |
| Malignant_lesions | 0.8807 | 0.9850 | 0.9627 | 0.9199 | 0.9494 | 0.9893 | 0.9819 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9870 | 0.9741 | 0.9766 | 0.9896 | 0.9989 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=20 TN=348 FN=6
- **Malignant_lesions**: TP=155 FP=6 TN=394 FN=21
- **body_of_stomach_normal**: TP=188 FP=5 TN=379 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 18 | 155 | 3 |
| body_of_stomach_normal | 2 | 2 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9099    0.9712    0.9395       208
     Malignant_lesions     0.9627    0.8807    0.9199       176
body_of_stomach_normal     0.9741    0.9792    0.9766       192

              accuracy                         0.9462       576
             macro avg     0.9489    0.9437    0.9453       576
          weighted avg     0.9474    0.9462    0.9459       576

```