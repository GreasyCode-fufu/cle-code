# mbv3_sft_b2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b2_seed46/mbv3_sft_b2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9366 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9915 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9370 |
| Balanced Acc | 0.9346 |
| Macro-Sensitivity | 0.9346 |
| Macro-Specificity | 0.9680 |
| Macro-Precision | 0.9419 |
| AUC-PR (Macro) | 0.9837 |
| Cohen's Kappa | 0.9058 |
| MCC | 0.9072 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 5124.5 (0.20 ms/张) |
| End-to-End FPS | 113.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9375 | 0.8969 | 0.9281 | 0.9773 | 0.9871 | 0.9774 | 208 |
| Malignant_lesions | 0.8580 | 0.9900 | 0.9742 | 0.9124 | 0.9406 | 0.9885 | 0.9759 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9766 | 0.9545 | 0.9692 | 0.9921 | 0.9988 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=23 TN=345 FN=8
- **Malignant_lesions**: TP=151 FP=4 TN=396 FN=25
- **body_of_stomach_normal**: TP=189 FP=9 TN=375 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 4 | 4 |
| Malignant_lesions | 20 | 151 | 5 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8969    0.9615    0.9281       208
     Malignant_lesions     0.9742    0.8580    0.9124       176
body_of_stomach_normal     0.9545    0.9844    0.9692       192

              accuracy                         0.9375       576
             macro avg     0.9419    0.9346    0.9366       576
          weighted avg     0.9397    0.9375    0.9370       576

```