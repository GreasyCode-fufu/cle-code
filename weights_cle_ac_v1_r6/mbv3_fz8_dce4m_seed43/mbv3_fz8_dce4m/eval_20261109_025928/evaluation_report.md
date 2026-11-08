# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz8_dce4m_seed43/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9585 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9948 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9584 |
| Balanced Acc | 0.9581 |
| Macro-Sensitivity | 0.9581 |
| Macro-Specificity | 0.9790 |
| Macro-Precision | 0.9589 |
| AUC-PR (Macro) | 0.9898 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9374 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3909.9 (0.26 ms/张) |
| End-to-End FPS | 17.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9647 | 0.9384 | 0.9451 | 0.9726 | 0.9924 | 0.9887 | 208 |
| Malignant_lesions | 0.9432 | 0.9800 | 0.9540 | 0.9486 | 0.9751 | 0.9927 | 0.9821 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9993 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=13 TN=355 FN=10
- **Malignant_lesions**: TP=166 FP=8 TN=392 FN=10
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 10 | 166 | 0 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9384    0.9519    0.9451       208
     Malignant_lesions     0.9540    0.9432    0.9486       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9583       576
             macro avg     0.9589    0.9581    0.9585       576
          weighted avg     0.9585    0.9583    0.9584       576

```