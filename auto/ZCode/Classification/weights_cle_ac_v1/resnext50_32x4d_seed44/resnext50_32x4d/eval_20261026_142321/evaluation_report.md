# resnext50_32x4d 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/resnext50_32x4d_seed44/resnext50_32x4d/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9441 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9927 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9444 |
| Balanced Acc | 0.9415 |
| Macro-Sensitivity | 0.9415 |
| Macro-Specificity | 0.9714 |
| Macro-Precision | 0.9499 |
| AUC-PR (Macro) | 0.9863 |
| Cohen's Kappa | 0.9162 |
| MCC | 0.9179 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.29G |
| FLOPs | 8.57G |
| Params | 22.99M |
| Pure-Inference FPS | 1637.8 (0.61 ms/张) |
| End-to-End FPS | 255.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9321 | 0.8908 | 0.9336 | 0.9885 | 0.9903 | 0.9830 | 208 |
| Malignant_lesions | 0.8750 | 0.9900 | 0.9747 | 0.9222 | 0.9474 | 0.9884 | 0.9773 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9922 | 0.9841 | 0.9764 | 0.9845 | 0.9993 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=25 TN=343 FN=4
- **Malignant_lesions**: TP=154 FP=4 TN=396 FN=22
- **body_of_stomach_normal**: TP=186 FP=3 TN=381 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 4 | 0 |
| Malignant_lesions | 19 | 154 | 3 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8908    0.9808    0.9336       208
     Malignant_lesions     0.9747    0.8750    0.9222       176
body_of_stomach_normal     0.9841    0.9688    0.9764       192

              accuracy                         0.9444       576
             macro avg     0.9499    0.9415    0.9441       576
          weighted avg     0.9476    0.9444    0.9444       576

```