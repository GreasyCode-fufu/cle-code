# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz8m_seed43/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9375 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9937 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9376 |
| Balanced Acc | 0.9342 |
| Macro-Sensitivity | 0.9342 |
| Macro-Specificity | 0.9677 |
| Macro-Precision | 0.9455 |
| AUC-PR (Macro) | 0.9874 |
| Cohen's Kappa | 0.9057 |
| MCC | 0.9081 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2711.5 (0.37 ms/张) |
| End-to-End FPS | 152.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9185 | 0.8718 | 0.9231 | 0.9883 | 0.9913 | 0.9869 | 208 |
| Malignant_lesions | 0.8636 | 0.9925 | 0.9806 | 0.9184 | 0.9430 | 0.9914 | 0.9782 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9922 | 0.9840 | 0.9710 | 0.9794 | 0.9985 | 0.9972 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=30 TN=338 FN=4
- **Malignant_lesions**: TP=152 FP=3 TN=397 FN=24
- **body_of_stomach_normal**: TP=184 FP=3 TN=381 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 2 | 2 |
| Malignant_lesions | 23 | 152 | 1 |
| body_of_stomach_normal | 7 | 1 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8718    0.9808    0.9231       208
     Malignant_lesions     0.9806    0.8636    0.9184       176
body_of_stomach_normal     0.9840    0.9583    0.9710       192

              accuracy                         0.9375       576
             macro avg     0.9455    0.9342    0.9375       576
          weighted avg     0.9424    0.9375    0.9376       576

```