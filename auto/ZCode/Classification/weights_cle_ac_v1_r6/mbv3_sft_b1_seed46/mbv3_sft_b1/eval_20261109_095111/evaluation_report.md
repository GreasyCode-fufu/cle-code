# mbv3_sft_b1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b1_seed46/mbv3_sft_b1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8902 |
| Accuracy | 0.8924 |
| Macro-AUROC (OvR) | 0.9820 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8910 |
| Balanced Acc | 0.8882 |
| Macro-Sensitivity | 0.8882 |
| Macro-Specificity | 0.9450 |
| Macro-Precision | 0.8999 |
| AUC-PR (Macro) | 0.9663 |
| Cohen's Kappa | 0.8375 |
| MCC | 0.8407 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3714.3 (0.27 ms/张) |
| End-to-End FPS | 74.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9183 | 0.8913 | 0.8268 | 0.8702 | 0.9507 | 0.9698 | 0.9503 | 208 |
| Malignant_lesions | 0.7670 | 0.9775 | 0.9375 | 0.8438 | 0.9051 | 0.9794 | 0.9544 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9661 | 0.9353 | 0.9567 | 0.9893 | 0.9968 | 0.9941 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=40 TN=328 FN=17
- **Malignant_lesions**: TP=135 FP=9 TN=391 FN=41
- **body_of_stomach_normal**: TP=188 FP=13 TN=371 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 9 | 8 |
| Malignant_lesions | 36 | 135 | 5 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8268    0.9183    0.8702       208
     Malignant_lesions     0.9375    0.7670    0.8438       176
body_of_stomach_normal     0.9353    0.9792    0.9567       192

              accuracy                         0.8924       576
             macro avg     0.8999    0.8882    0.8902       576
          weighted avg     0.8968    0.8924    0.8910       576

```