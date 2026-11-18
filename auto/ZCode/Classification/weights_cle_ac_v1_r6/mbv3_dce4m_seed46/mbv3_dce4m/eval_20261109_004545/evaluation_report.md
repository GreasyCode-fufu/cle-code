# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce4m_seed46/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9553 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9914 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9550 |
| Balanced Acc | 0.9545 |
| Macro-Sensitivity | 0.9545 |
| Macro-Specificity | 0.9771 |
| Macro-Precision | 0.9564 |
| AUC-PR (Macro) | 0.9837 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9322 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5629.9 (0.18 ms/张) |
| End-to-End FPS | 229.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9565 | 0.9256 | 0.9409 | 0.9751 | 0.9876 | 0.9779 | 208 |
| Malignant_lesions | 0.9432 | 0.9825 | 0.9595 | 0.9513 | 0.9752 | 0.9894 | 0.9780 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9922 | 0.9840 | 0.9737 | 0.9820 | 0.9971 | 0.9953 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=16 TN=352 FN=9
- **Malignant_lesions**: TP=166 FP=7 TN=393 FN=10
- **body_of_stomach_normal**: TP=185 FP=3 TN=381 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 9 | 166 | 1 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9256    0.9567    0.9409       208
     Malignant_lesions     0.9595    0.9432    0.9513       176
body_of_stomach_normal     0.9840    0.9635    0.9737       192

              accuracy                         0.9549       576
             macro avg     0.9564    0.9545    0.9553       576
          weighted avg     0.9554    0.9549    0.9550       576

```