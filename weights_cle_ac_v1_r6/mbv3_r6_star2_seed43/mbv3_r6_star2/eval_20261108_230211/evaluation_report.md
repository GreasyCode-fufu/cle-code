# mbv3_r6_star2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star2_seed43/mbv3_r6_star2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9497 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9924 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9495 |
| Balanced Acc | 0.9493 |
| Macro-Sensitivity | 0.9493 |
| Macro-Specificity | 0.9746 |
| Macro-Precision | 0.9502 |
| AUC-PR (Macro) | 0.9859 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9244 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 3652.7 (0.27 ms/张) |
| End-to-End FPS | 223.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9620 | 0.9330 | 0.9353 | 0.9646 | 0.9881 | 0.9811 | 208 |
| Malignant_lesions | 0.9261 | 0.9800 | 0.9532 | 0.9395 | 0.9679 | 0.9915 | 0.9828 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9818 | 0.9643 | 0.9742 | 0.9921 | 0.9976 | 0.9937 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=14 TN=354 FN=13
- **Malignant_lesions**: TP=163 FP=8 TN=392 FN=13
- **body_of_stomach_normal**: TP=189 FP=7 TN=377 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 8 | 5 |
| Malignant_lesions | 11 | 163 | 2 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9330    0.9375    0.9353       208
     Malignant_lesions     0.9532    0.9261    0.9395       176
body_of_stomach_normal     0.9643    0.9844    0.9742       192

              accuracy                         0.9497       576
             macro avg     0.9502    0.9493    0.9497       576
          weighted avg     0.9496    0.9497    0.9495       576

```