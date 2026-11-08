# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz8m_seed43/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9582 |
| Accuracy | 0.9672 |
| Macro-AUROC (OvR) | 0.9977 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9672 |
| Balanced Acc | 0.9585 |
| Macro-Sensitivity | 0.9585 |
| Macro-Specificity | 0.9835 |
| Macro-Precision | 0.9580 |
| AUC-PR (Macro) | 0.9928 |
| Cohen's Kappa | 0.9479 |
| MCC | 0.9479 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2347.9 (0.43 ms/张) |
| End-to-End FPS | 190.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9641 | 0.9741 | 0.9691 | 0.9666 | 0.9700 | 0.9977 | 0.9973 | 195 |
| Malignant_lesions | 0.9114 | 0.9799 | 0.9114 | 0.9114 | 0.9799 | 0.9955 | 0.9811 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=6 TN=226 FN=7
- **Malignant_lesions**: TP=72 FP=7 TN=341 FN=7
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 7 | 0 |
| Malignant_lesions | 6 | 72 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9691    0.9641    0.9666       195
     Malignant_lesions     0.9114    0.9114    0.9114        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9672       427
             macro avg     0.9580    0.9585    0.9582       427
          weighted avg     0.9672    0.9672    0.9672       427

```