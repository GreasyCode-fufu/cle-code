# starnet_s2_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_seed45/starnet_s2_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9750 |
| Accuracy | 0.9813 |
| Macro-AUROC (OvR) | 0.9974 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9662 |
| Macro-Sensitivity | 0.9662 |
| Macro-Specificity | 0.9889 |
| Macro-Precision | 0.9857 |
| AUC-PR (Macro) | 0.9928 |
| Cohen's Kappa | 0.9700 |
| MCC | 0.9705 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 2623.9 (0.38 ms/张) |
| End-to-End FPS | 121.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 1.0000 | 0.9741 | 0.9701 | 0.9848 | 1.0000 | 0.9976 | 0.9970 | 195 |
| Malignant_lesions | 0.8987 | 1.0000 | 1.0000 | 0.9467 | 0.9775 | 0.9947 | 0.9815 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 1.0000 | 0.9999 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=6 TN=226 FN=0
- **Malignant_lesions**: TP=71 FP=0 TN=348 FN=8
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 0 | 0 |
| Malignant_lesions | 6 | 71 | 2 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9701    1.0000    0.9848       195
     Malignant_lesions     1.0000    0.8987    0.9467        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9813       427
             macro avg     0.9857    0.9662    0.9750       427
          weighted avg     0.9817    0.9813    0.9809       427

```