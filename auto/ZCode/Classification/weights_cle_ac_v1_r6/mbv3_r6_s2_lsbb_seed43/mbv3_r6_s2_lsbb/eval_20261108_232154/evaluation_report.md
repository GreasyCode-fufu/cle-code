# mbv3_r6_s2_lsbb 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_lsbb_seed43/mbv3_r6_s2_lsbb/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9384 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9927 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9389 |
| Balanced Acc | 0.9381 |
| Macro-Sensitivity | 0.9381 |
| Macro-Specificity | 0.9694 |
| Macro-Precision | 0.9392 |
| AUC-PR (Macro) | 0.9855 |
| Cohen's Kappa | 0.9086 |
| MCC | 0.9088 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 235.94M |
| FLOPs | 471.88M |
| Params | 3.27M |
| Pure-Inference FPS | 3125.4 (0.32 ms/张) |
| End-to-End FPS | 164.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9592 | 0.9282 | 0.9305 | 0.9619 | 0.9899 | 0.9830 | 208 |
| Malignant_lesions | 0.8920 | 0.9725 | 0.9345 | 0.9128 | 0.9534 | 0.9889 | 0.9749 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9766 | 0.9548 | 0.9719 | 0.9947 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=15 TN=353 FN=14
- **Malignant_lesions**: TP=157 FP=11 TN=389 FN=19
- **body_of_stomach_normal**: TP=190 FP=9 TN=375 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 11 | 3 |
| Malignant_lesions | 13 | 157 | 6 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9282    0.9327    0.9305       208
     Malignant_lesions     0.9345    0.8920    0.9128       176
body_of_stomach_normal     0.9548    0.9896    0.9719       192

              accuracy                         0.9392       576
             macro avg     0.9392    0.9381    0.9384       576
          weighted avg     0.9390    0.9392    0.9389       576

```