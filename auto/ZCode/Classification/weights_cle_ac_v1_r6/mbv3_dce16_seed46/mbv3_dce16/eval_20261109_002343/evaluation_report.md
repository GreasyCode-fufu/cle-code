# mbv3_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce16_seed46/mbv3_dce16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9500 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9896 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9498 |
| Balanced Acc | 0.9497 |
| Macro-Sensitivity | 0.9497 |
| Macro-Specificity | 0.9746 |
| Macro-Precision | 0.9506 |
| AUC-PR (Macro) | 0.9786 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9244 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 5220.4 (0.19 ms/张) |
| End-to-End FPS | 11.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9565 | 0.9245 | 0.9333 | 0.9670 | 0.9842 | 0.9732 | 208 |
| Malignant_lesions | 0.9432 | 0.9725 | 0.9379 | 0.9405 | 0.9749 | 0.9868 | 0.9663 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9948 | 0.9893 | 0.9763 | 0.9820 | 0.9977 | 0.9962 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=16 TN=352 FN=12
- **Malignant_lesions**: TP=166 FP=11 TN=389 FN=10
- **body_of_stomach_normal**: TP=185 FP=2 TN=382 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 11 | 1 |
| Malignant_lesions | 9 | 166 | 1 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9245    0.9423    0.9333       208
     Malignant_lesions     0.9379    0.9432    0.9405       176
body_of_stomach_normal     0.9893    0.9635    0.9763       192

              accuracy                         0.9497       576
             macro avg     0.9506    0.9497    0.9500       576
          weighted avg     0.9502    0.9497    0.9498       576

```