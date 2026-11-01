# convnext_cmp_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_cmp_fgc_seed43/convnext_cmp_fgc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9531 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9954 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9533 |
| Balanced Acc | 0.9522 |
| Macro-Sensitivity | 0.9522 |
| Macro-Specificity | 0.9763 |
| Macro-Precision | 0.9545 |
| AUC-PR (Macro) | 0.9916 |
| Cohen's Kappa | 0.9295 |
| MCC | 0.9298 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 28.14M |
| Pure-Inference FPS | 465.8 (2.15 ms/张) |
| End-to-End FPS | 183.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9565 | 0.9263 | 0.9459 | 0.9805 | 0.9923 | 0.9899 | 208 |
| Malignant_lesions | 0.9318 | 0.9750 | 0.9425 | 0.9371 | 0.9701 | 0.9952 | 0.9877 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9974 | 0.9946 | 0.9761 | 0.9795 | 0.9986 | 0.9971 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=16 TN=352 FN=7
- **Malignant_lesions**: TP=164 FP=10 TN=390 FN=12
- **body_of_stomach_normal**: TP=184 FP=1 TN=383 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 12 | 164 | 0 |
| body_of_stomach_normal | 4 | 4 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9263    0.9663    0.9459       208
     Malignant_lesions     0.9425    0.9318    0.9371       176
body_of_stomach_normal     0.9946    0.9583    0.9761       192

              accuracy                         0.9531       576
             macro avg     0.9545    0.9522    0.9531       576
          weighted avg     0.9540    0.9531    0.9533       576

```