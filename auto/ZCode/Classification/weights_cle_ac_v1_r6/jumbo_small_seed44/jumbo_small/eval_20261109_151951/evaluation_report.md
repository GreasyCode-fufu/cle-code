# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/jumbo_small_seed44/jumbo_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9597 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9946 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9599 |
| Balanced Acc | 0.9599 |
| Macro-Sensitivity | 0.9599 |
| Macro-Specificity | 0.9800 |
| Macro-Precision | 0.9595 |
| AUC-PR (Macro) | 0.9893 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9400 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 969.7 (1.03 ms/张) |
| End-to-End FPS | 215.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9755 | 0.9561 | 0.9492 | 0.9677 | 0.9924 | 0.9871 | 208 |
| Malignant_lesions | 0.9375 | 0.9750 | 0.9429 | 0.9402 | 0.9726 | 0.9916 | 0.9812 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9896 | 0.9796 | 0.9897 | 1.0000 | 0.9999 | 0.9997 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=9 TN=359 FN=12
- **Malignant_lesions**: TP=165 FP=10 TN=390 FN=11
- **body_of_stomach_normal**: TP=192 FP=4 TN=380 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 10 | 2 |
| Malignant_lesions | 9 | 165 | 2 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9561    0.9423    0.9492       208
     Malignant_lesions     0.9429    0.9375    0.9402       176
body_of_stomach_normal     0.9796    1.0000    0.9897       192

              accuracy                         0.9601       576
             macro avg     0.9595    0.9599    0.9597       576
          weighted avg     0.9599    0.9601    0.9599       576

```