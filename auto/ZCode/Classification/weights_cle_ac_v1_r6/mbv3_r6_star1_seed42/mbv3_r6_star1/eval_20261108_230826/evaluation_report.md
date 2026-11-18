# mbv3_r6_star1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star1_seed42/mbv3_r6_star1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9496 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9924 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9496 |
| Balanced Acc | 0.9475 |
| Macro-Sensitivity | 0.9475 |
| Macro-Specificity | 0.9741 |
| Macro-Precision | 0.9537 |
| AUC-PR (Macro) | 0.9863 |
| Cohen's Kappa | 0.9241 |
| MCC | 0.9252 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.18M |
| FLOPs | 464.35M |
| Params | 3.47M |
| Pure-Inference FPS | 3598.2 (0.28 ms/张) |
| End-to-End FPS | 227.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9402 | 0.9018 | 0.9352 | 0.9830 | 0.9876 | 0.9813 | 208 |
| Malignant_lesions | 0.8920 | 0.9900 | 0.9752 | 0.9318 | 0.9542 | 0.9909 | 0.9803 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9988 | 0.9975 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=22 TN=346 FN=6
- **Malignant_lesions**: TP=157 FP=4 TN=396 FN=19
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 18 | 157 | 1 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9018    0.9712    0.9352       208
     Malignant_lesions     0.9752    0.8920    0.9318       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9497       576
             macro avg     0.9537    0.9475    0.9496       576
          weighted avg     0.9517    0.9497    0.9496       576

```