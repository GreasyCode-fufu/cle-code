# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_seed45/fast_t0/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9161 |
| Accuracy | 0.9184 |
| Macro-AUROC (OvR) | 0.9900 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9171 |
| Balanced Acc | 0.9142 |
| Macro-Sensitivity | 0.9142 |
| Macro-Specificity | 0.9583 |
| Macro-Precision | 0.9240 |
| AUC-PR (Macro) | 0.9800 |
| Cohen's Kappa | 0.8769 |
| MCC | 0.8794 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 4512.7 (0.22 ms/张) |
| End-to-End FPS | 197.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9158 | 0.8646 | 0.9062 | 0.9712 | 0.9853 | 0.9755 | 208 |
| Malignant_lesions | 0.8011 | 0.9825 | 0.9527 | 0.8704 | 0.9182 | 0.9854 | 0.9661 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9766 | 0.9548 | 0.9719 | 0.9947 | 0.9992 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=31 TN=337 FN=10
- **Malignant_lesions**: TP=141 FP=7 TN=393 FN=35
- **body_of_stomach_normal**: TP=190 FP=9 TN=375 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 29 | 141 | 6 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8646    0.9519    0.9062       208
     Malignant_lesions     0.9527    0.8011    0.8704       176
body_of_stomach_normal     0.9548    0.9896    0.9719       192

              accuracy                         0.9184       576
             macro avg     0.9240    0.9142    0.9161       576
          weighted avg     0.9216    0.9184    0.9171       576

```