# mbv3_sft_b1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b1_seed42/mbv3_sft_b1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8976 |
| Accuracy | 0.8993 |
| Macro-AUROC (OvR) | 0.9832 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8980 |
| Balanced Acc | 0.8966 |
| Macro-Sensitivity | 0.8966 |
| Macro-Specificity | 0.9488 |
| Macro-Precision | 0.9036 |
| AUC-PR (Macro) | 0.9691 |
| Cohen's Kappa | 0.8482 |
| MCC | 0.8503 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3908.1 (0.26 ms/张) |
| End-to-End FPS | 107.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8990 | 0.9185 | 0.8618 | 0.8800 | 0.9415 | 0.9724 | 0.9553 | 208 |
| Malignant_lesions | 0.8011 | 0.9775 | 0.9400 | 0.8650 | 0.9178 | 0.9810 | 0.9597 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9505 | 0.9091 | 0.9476 | 0.9946 | 0.9960 | 0.9923 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=30 TN=338 FN=21
- **Malignant_lesions**: TP=141 FP=9 TN=391 FN=35
- **body_of_stomach_normal**: TP=190 FP=19 TN=365 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 9 | 12 |
| Malignant_lesions | 28 | 141 | 7 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8618    0.8990    0.8800       208
     Malignant_lesions     0.9400    0.8011    0.8650       176
body_of_stomach_normal     0.9091    0.9896    0.9476       192

              accuracy                         0.8993       576
             macro avg     0.9036    0.8966    0.8976       576
          weighted avg     0.9014    0.8993    0.8980       576

```