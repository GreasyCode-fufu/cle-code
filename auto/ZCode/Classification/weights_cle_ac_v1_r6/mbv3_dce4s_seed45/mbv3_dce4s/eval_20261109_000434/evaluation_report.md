# mbv3_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce4s_seed45/mbv3_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9499 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9917 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9498 |
| Balanced Acc | 0.9495 |
| Macro-Sensitivity | 0.9495 |
| Macro-Specificity | 0.9746 |
| Macro-Precision | 0.9504 |
| AUC-PR (Macro) | 0.9840 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9244 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3882.0 (0.26 ms/张) |
| End-to-End FPS | 10.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9592 | 0.9292 | 0.9381 | 0.9698 | 0.9875 | 0.9781 | 208 |
| Malignant_lesions | 0.9432 | 0.9725 | 0.9379 | 0.9405 | 0.9749 | 0.9887 | 0.9762 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9922 | 0.9840 | 0.9710 | 0.9794 | 0.9987 | 0.9976 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=15 TN=353 FN=11
- **Malignant_lesions**: TP=166 FP=11 TN=389 FN=10
- **body_of_stomach_normal**: TP=184 FP=3 TN=381 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 9 | 166 | 1 |
| body_of_stomach_normal | 6 | 2 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9292    0.9471    0.9381       208
     Malignant_lesions     0.9379    0.9432    0.9405       176
body_of_stomach_normal     0.9840    0.9583    0.9710       192

              accuracy                         0.9497       576
             macro avg     0.9504    0.9495    0.9499       576
          weighted avg     0.9501    0.9497    0.9498       576

```