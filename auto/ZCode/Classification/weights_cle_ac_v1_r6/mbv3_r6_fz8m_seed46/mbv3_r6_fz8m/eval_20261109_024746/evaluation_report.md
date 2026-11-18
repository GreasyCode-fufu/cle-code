# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz8m_seed46/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9710 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9973 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9768 |
| Balanced Acc | 0.9779 |
| Macro-Sensitivity | 0.9779 |
| Macro-Specificity | 0.9895 |
| Macro-Precision | 0.9651 |
| AUC-PR (Macro) | 0.9916 |
| Cohen's Kappa | 0.9630 |
| MCC | 0.9633 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2831.4 (0.35 ms/张) |
| End-to-End FPS | 234.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9914 | 0.9894 | 0.9740 | 0.9664 | 0.9971 | 0.9965 | 195 |
| Malignant_lesions | 0.9747 | 0.9770 | 0.9059 | 0.9390 | 0.9942 | 0.9947 | 0.9785 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=2 TN=230 FN=8
- **Malignant_lesions**: TP=77 FP=8 TN=340 FN=2
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 8 | 0 |
| Malignant_lesions | 2 | 77 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9894    0.9590    0.9740       195
     Malignant_lesions     0.9059    0.9747    0.9390        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9766       427
             macro avg     0.9651    0.9779    0.9710       427
          weighted avg     0.9778    0.9766    0.9768       427

```