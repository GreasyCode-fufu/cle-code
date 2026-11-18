# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/resnet50_seed43/resnet50/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9439 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9927 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9442 |
| Balanced Acc | 0.9441 |
| Macro-Sensitivity | 0.9441 |
| Macro-Specificity | 0.9722 |
| Macro-Precision | 0.9441 |
| AUC-PR (Macro) | 0.9857 |
| Cohen's Kappa | 0.9165 |
| MCC | 0.9166 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 866.9 (1.15 ms/张) |
| End-to-End FPS | 217.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9674 | 0.9415 | 0.9346 | 0.9596 | 0.9899 | 0.9814 | 208 |
| Malignant_lesions | 0.9148 | 0.9725 | 0.9360 | 0.9253 | 0.9629 | 0.9887 | 0.9769 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9766 | 0.9548 | 0.9719 | 0.9947 | 0.9993 | 0.9988 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=12 TN=356 FN=15
- **Malignant_lesions**: TP=161 FP=11 TN=389 FN=15
- **body_of_stomach_normal**: TP=190 FP=9 TN=375 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 11 | 4 |
| Malignant_lesions | 10 | 161 | 5 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9415    0.9279    0.9346       208
     Malignant_lesions     0.9360    0.9148    0.9253       176
body_of_stomach_normal     0.9548    0.9896    0.9719       192

              accuracy                         0.9444       576
             macro avg     0.9441    0.9441    0.9439       576
          weighted avg     0.9442    0.9444    0.9442       576

```