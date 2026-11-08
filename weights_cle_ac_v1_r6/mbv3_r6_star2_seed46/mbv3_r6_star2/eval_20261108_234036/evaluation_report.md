# mbv3_r6_star2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star2_seed46/mbv3_r6_star2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9649 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9973 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9721 |
| Balanced Acc | 0.9694 |
| Macro-Sensitivity | 0.9694 |
| Macro-Specificity | 0.9866 |
| Macro-Precision | 0.9609 |
| AUC-PR (Macro) | 0.9911 |
| Cohen's Kappa | 0.9555 |
| MCC | 0.9556 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 4092.5 (0.24 ms/张) |
| End-to-End FPS | 282.7 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9828 | 0.9791 | 0.9689 | 0.9661 | 0.9971 | 0.9967 | 195 |
| Malignant_lesions | 0.9494 | 0.9770 | 0.9036 | 0.9259 | 0.9884 | 0.9947 | 0.9767 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=4 TN=228 FN=8
- **Malignant_lesions**: TP=75 FP=8 TN=340 FN=4
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 8 | 0 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9791    0.9590    0.9689       195
     Malignant_lesions     0.9036    0.9494    0.9259        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9719       427
             macro avg     0.9609    0.9694    0.9649       427
          weighted avg     0.9726    0.9719    0.9721       427

```