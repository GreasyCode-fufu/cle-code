# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz16_seed43/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9519 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9959 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9516 |
| Balanced Acc | 0.9507 |
| Macro-Sensitivity | 0.9507 |
| Macro-Specificity | 0.9752 |
| Macro-Precision | 0.9537 |
| AUC-PR (Macro) | 0.9917 |
| Cohen's Kappa | 0.9268 |
| MCC | 0.9272 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4240.6 (0.24 ms/张) |
| End-to-End FPS | 63.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9511 | 0.9174 | 0.9390 | 0.9777 | 0.9938 | 0.9914 | 208 |
| Malignant_lesions | 0.9375 | 0.9850 | 0.9649 | 0.9510 | 0.9728 | 0.9959 | 0.9871 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9896 | 0.9786 | 0.9657 | 0.9769 | 0.9981 | 0.9965 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=18 TN=350 FN=8
- **Malignant_lesions**: TP=165 FP=6 TN=394 FN=11
- **body_of_stomach_normal**: TP=183 FP=4 TN=380 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 9 | 165 | 2 |
| body_of_stomach_normal | 9 | 0 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9174    0.9615    0.9390       208
     Malignant_lesions     0.9649    0.9375    0.9510       176
body_of_stomach_normal     0.9786    0.9531    0.9657       192

              accuracy                         0.9514       576
             macro avg     0.9537    0.9507    0.9519       576
          weighted avg     0.9523    0.9514    0.9516       576

```