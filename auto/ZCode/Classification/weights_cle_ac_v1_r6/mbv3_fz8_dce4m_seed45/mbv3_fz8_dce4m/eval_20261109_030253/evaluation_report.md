# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz8_dce4m_seed45/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9583 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9952 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9584 |
| Balanced Acc | 0.9590 |
| Macro-Sensitivity | 0.9590 |
| Macro-Specificity | 0.9793 |
| Macro-Precision | 0.9578 |
| AUC-PR (Macro) | 0.9913 |
| Cohen's Kappa | 0.9374 |
| MCC | 0.9375 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4990.9 (0.20 ms/张) |
| End-to-End FPS | 14.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9783 | 0.9606 | 0.9489 | 0.9651 | 0.9920 | 0.9883 | 208 |
| Malignant_lesions | 0.9602 | 0.9675 | 0.9286 | 0.9441 | 0.9822 | 0.9951 | 0.9893 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9984 | 0.9964 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=8 TN=360 FN=13
- **Malignant_lesions**: TP=169 FP=13 TN=387 FN=7
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 10 | 3 |
| Malignant_lesions | 7 | 169 | 0 |
| body_of_stomach_normal | 1 | 3 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9606    0.9375    0.9489       208
     Malignant_lesions     0.9286    0.9602    0.9441       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9583       576
             macro avg     0.9578    0.9590    0.9583       576
          weighted avg     0.9587    0.9583    0.9584       576

```