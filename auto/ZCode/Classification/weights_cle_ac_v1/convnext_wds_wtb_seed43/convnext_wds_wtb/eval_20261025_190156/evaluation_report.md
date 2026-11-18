# convnext_wds_wtb 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wtb_seed43/convnext_wds_wtb/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9476 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9938 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9479 |
| Balanced Acc | 0.9476 |
| Macro-Sensitivity | 0.9476 |
| Macro-Specificity | 0.9739 |
| Macro-Precision | 0.9476 |
| AUC-PR (Macro) | 0.9879 |
| Cohen's Kappa | 0.9217 |
| MCC | 0.9217 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.53G |
| FLOPs | 9.07G |
| Params | 29.11M |
| Pure-Inference FPS | 514.3 (1.94 ms/张) |
| End-to-End FPS | 195.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9620 | 0.9327 | 0.9327 | 0.9620 | 0.9915 | 0.9859 | 208 |
| Malignant_lesions | 0.9205 | 0.9675 | 0.9257 | 0.9231 | 0.9651 | 0.9904 | 0.9785 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9922 | 0.9845 | 0.9870 | 0.9948 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=14 TN=354 FN=14
- **Malignant_lesions**: TP=162 FP=13 TN=387 FN=14
- **body_of_stomach_normal**: TP=190 FP=3 TN=381 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 13 | 1 |
| Malignant_lesions | 12 | 162 | 2 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9327    0.9327    0.9327       208
     Malignant_lesions     0.9257    0.9205    0.9231       176
body_of_stomach_normal     0.9845    0.9896    0.9870       192

              accuracy                         0.9479       576
             macro avg     0.9476    0.9476    0.9476       576
          weighted avg     0.9478    0.9479    0.9479       576

```