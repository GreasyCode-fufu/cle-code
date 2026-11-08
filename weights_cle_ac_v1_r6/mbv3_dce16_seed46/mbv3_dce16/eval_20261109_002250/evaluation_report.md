# mbv3_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce16_seed46/mbv3_dce16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9588 |
| Accuracy | 0.9672 |
| Macro-AUROC (OvR) | 0.9970 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9673 |
| Balanced Acc | 0.9610 |
| Macro-Sensitivity | 0.9610 |
| Macro-Specificity | 0.9837 |
| Macro-Precision | 0.9567 |
| AUC-PR (Macro) | 0.9902 |
| Cohen's Kappa | 0.9480 |
| MCC | 0.9480 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 3771.9 (0.27 ms/张) |
| End-to-End FPS | 15.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9741 | 0.9689 | 0.9639 | 0.9658 | 0.9969 | 0.9965 | 195 |
| Malignant_lesions | 0.9241 | 0.9770 | 0.9012 | 0.9125 | 0.9827 | 0.9940 | 0.9742 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=6 TN=226 FN=8
- **Malignant_lesions**: TP=73 FP=8 TN=340 FN=6
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 8 | 0 |
| Malignant_lesions | 6 | 73 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9689    0.9590    0.9639       195
     Malignant_lesions     0.9012    0.9241    0.9125        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9672       427
             macro avg     0.9567    0.9610    0.9588       427
          weighted avg     0.9675    0.9672    0.9673       427

```