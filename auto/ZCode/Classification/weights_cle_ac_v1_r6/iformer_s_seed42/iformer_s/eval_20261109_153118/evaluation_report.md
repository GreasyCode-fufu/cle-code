# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/iformer_s_seed42/iformer_s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9550 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9925 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9548 |
| Balanced Acc | 0.9548 |
| Macro-Sensitivity | 0.9548 |
| Macro-Specificity | 0.9772 |
| Macro-Precision | 0.9553 |
| AUC-PR (Macro) | 0.9869 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9321 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 2197.2 (0.46 ms/张) |
| End-to-End FPS | 237.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9674 | 0.9426 | 0.9448 | 0.9700 | 0.9903 | 0.9830 | 208 |
| Malignant_lesions | 0.9432 | 0.9825 | 0.9595 | 0.9513 | 0.9752 | 0.9896 | 0.9816 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9818 | 0.9639 | 0.9689 | 0.9869 | 0.9977 | 0.9962 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=12 TN=356 FN=11
- **Malignant_lesions**: TP=166 FP=7 TN=393 FN=10
- **body_of_stomach_normal**: TP=187 FP=7 TN=377 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 7 | 4 |
| Malignant_lesions | 7 | 166 | 3 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9426    0.9471    0.9448       208
     Malignant_lesions     0.9595    0.9432    0.9513       176
body_of_stomach_normal     0.9639    0.9740    0.9689       192

              accuracy                         0.9549       576
             macro avg     0.9553    0.9548    0.9550       576
          weighted avg     0.9549    0.9549    0.9548       576

```