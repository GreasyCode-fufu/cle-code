# convnext_msf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_msf_seed42/convnext_msf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9598 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9952 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9600 |
| Balanced Acc | 0.9600 |
| Macro-Sensitivity | 0.9600 |
| Macro-Specificity | 0.9800 |
| Macro-Precision | 0.9596 |
| AUC-PR (Macro) | 0.9907 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9400 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.48G |
| FLOPs | 8.95G |
| Params | 28.24M |
| Pure-Inference FPS | 805.9 (1.24 ms/张) |
| End-to-End FPS | 226.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9755 | 0.9563 | 0.9517 | 0.9703 | 0.9933 | 0.9880 | 208 |
| Malignant_lesions | 0.9432 | 0.9750 | 0.9432 | 0.9432 | 0.9750 | 0.9930 | 0.9852 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9896 | 0.9794 | 0.9845 | 0.9948 | 0.9993 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=9 TN=359 FN=11
- **Malignant_lesions**: TP=166 FP=10 TN=390 FN=10
- **body_of_stomach_normal**: TP=190 FP=4 TN=380 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 7 | 166 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9563    0.9471    0.9517       208
     Malignant_lesions     0.9432    0.9432    0.9432       176
body_of_stomach_normal     0.9794    0.9896    0.9845       192

              accuracy                         0.9601       576
             macro avg     0.9596    0.9600    0.9598       576
          weighted avg     0.9600    0.9601    0.9600       576

```