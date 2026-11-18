# mbv3_r6_star2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star2_seed42/mbv3_r6_star2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9478 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9922 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9480 |
| Balanced Acc | 0.9454 |
| Macro-Sensitivity | 0.9454 |
| Macro-Specificity | 0.9732 |
| Macro-Precision | 0.9529 |
| AUC-PR (Macro) | 0.9850 |
| Cohen's Kappa | 0.9215 |
| MCC | 0.9228 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 3594.1 (0.28 ms/张) |
| End-to-End FPS | 233.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9348 | 0.8943 | 0.9333 | 0.9857 | 0.9897 | 0.9830 | 208 |
| Malignant_lesions | 0.8864 | 0.9900 | 0.9750 | 0.9286 | 0.9519 | 0.9874 | 0.9730 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9948 | 0.9894 | 0.9816 | 0.9871 | 0.9994 | 0.9989 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=24 TN=344 FN=5
- **Malignant_lesions**: TP=156 FP=4 TN=396 FN=20
- **body_of_stomach_normal**: TP=187 FP=2 TN=382 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 3 | 2 |
| Malignant_lesions | 20 | 156 | 0 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8943    0.9760    0.9333       208
     Malignant_lesions     0.9750    0.8864    0.9286       176
body_of_stomach_normal     0.9894    0.9740    0.9816       192

              accuracy                         0.9479       576
             macro avg     0.9529    0.9454    0.9478       576
          weighted avg     0.9507    0.9479    0.9480       576

```