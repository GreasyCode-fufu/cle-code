# mbv3_r6_star2x 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star2x_seed42/mbv3_r6_star2x/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9262 |
| Accuracy | 0.9271 |
| Macro-AUROC (OvR) | 0.9888 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9267 |
| Balanced Acc | 0.9230 |
| Macro-Sensitivity | 0.9230 |
| Macro-Specificity | 0.9624 |
| Macro-Precision | 0.9363 |
| AUC-PR (Macro) | 0.9779 |
| Cohen's Kappa | 0.8899 |
| MCC | 0.8931 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.10M |
| FLOPs | 464.20M |
| Params | 3.39M |
| Pure-Inference FPS | 4485.4 (0.22 ms/张) |
| End-to-End FPS | 225.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9049 | 0.8517 | 0.9054 | 0.9794 | 0.9813 | 0.9637 | 208 |
| Malignant_lesions | 0.8182 | 0.9900 | 0.9730 | 0.8889 | 0.9252 | 0.9856 | 0.9708 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9922 | 0.9844 | 0.9844 | 0.9922 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=35 TN=333 FN=7
- **Malignant_lesions**: TP=144 FP=4 TN=396 FN=32
- **body_of_stomach_normal**: TP=189 FP=3 TN=381 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 4 | 3 |
| Malignant_lesions | 32 | 144 | 0 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8517    0.9663    0.9054       208
     Malignant_lesions     0.9730    0.8182    0.8889       176
body_of_stomach_normal     0.9844    0.9844    0.9844       192

              accuracy                         0.9271       576
             macro avg     0.9363    0.9230    0.9262       576
          weighted avg     0.9330    0.9271    0.9267       576

```