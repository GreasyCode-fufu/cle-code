# mbv3_r6_s2x_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2x_fz16_seed43/mbv3_r6_s2x_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9459 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9933 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9461 |
| Balanced Acc | 0.9441 |
| Macro-Sensitivity | 0.9441 |
| Macro-Specificity | 0.9725 |
| Macro-Precision | 0.9493 |
| AUC-PR (Macro) | 0.9876 |
| Cohen's Kappa | 0.9189 |
| MCC | 0.9197 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.10M |
| FLOPs | 464.20M |
| Params | 3.39M |
| Pure-Inference FPS | 4834.5 (0.21 ms/张) |
| End-to-End FPS | 173.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9457 | 0.9099 | 0.9395 | 0.9831 | 0.9906 | 0.9846 | 208 |
| Malignant_lesions | 0.8977 | 0.9875 | 0.9693 | 0.9322 | 0.9564 | 0.9915 | 0.9821 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9844 | 0.9686 | 0.9661 | 0.9818 | 0.9979 | 0.9962 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=20 TN=348 FN=6
- **Malignant_lesions**: TP=158 FP=5 TN=395 FN=18
- **body_of_stomach_normal**: TP=185 FP=6 TN=378 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 14 | 158 | 4 |
| body_of_stomach_normal | 6 | 1 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9099    0.9712    0.9395       208
     Malignant_lesions     0.9693    0.8977    0.9322       176
body_of_stomach_normal     0.9686    0.9635    0.9661       192

              accuracy                         0.9462       576
             macro avg     0.9493    0.9441    0.9459       576
          weighted avg     0.9476    0.9462    0.9461       576

```