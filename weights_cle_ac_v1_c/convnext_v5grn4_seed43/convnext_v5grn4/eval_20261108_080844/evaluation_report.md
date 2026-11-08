# convnext_v5grn4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5grn4_seed43/convnext_v5grn4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9823 |
| Accuracy | 0.9859 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9860 |
| Balanced Acc | 0.9847 |
| Macro-Sensitivity | 0.9847 |
| Macro-Specificity | 0.9933 |
| Macro-Precision | 0.9801 |
| AUC-PR (Macro) | 0.9979 |
| Cohen's Kappa | 0.9777 |
| MCC | 0.9777 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1577.5 (0.63 ms/张) |
| End-to-End FPS | 263.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9914 | 0.9896 | 0.9845 | 0.9829 | 0.9992 | 0.9991 | 195 |
| Malignant_lesions | 0.9747 | 0.9885 | 0.9506 | 0.9625 | 0.9942 | 0.9987 | 0.9945 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=2 TN=230 FN=4
- **Malignant_lesions**: TP=77 FP=4 TN=344 FN=2
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 2 | 77 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9896    0.9795    0.9845       195
     Malignant_lesions     0.9506    0.9747    0.9625        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9859       427
             macro avg     0.9801    0.9847    0.9823       427
          weighted avg     0.9861    0.9859    0.9860       427

```