# fast_t0_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_dce_last_seed46/fast_t0_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9250 |
| Accuracy | 0.9253 |
| Macro-AUROC (OvR) | 0.9915 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9251 |
| Balanced Acc | 0.9240 |
| Macro-Sensitivity | 0.9240 |
| Macro-Specificity | 0.9622 |
| Macro-Precision | 0.9268 |
| AUC-PR (Macro) | 0.9837 |
| Cohen's Kappa | 0.8876 |
| MCC | 0.8880 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 5498.9 (0.18 ms/张) |
| End-to-End FPS | 8.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9375 | 0.8930 | 0.9078 | 0.9557 | 0.9864 | 0.9794 | 208 |
| Malignant_lesions | 0.8750 | 0.9725 | 0.9333 | 0.9032 | 0.9465 | 0.9901 | 0.9750 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9766 | 0.9541 | 0.9639 | 0.9868 | 0.9981 | 0.9966 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=23 TN=345 FN=16
- **Malignant_lesions**: TP=154 FP=11 TN=389 FN=22
- **body_of_stomach_normal**: TP=187 FP=9 TN=375 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 11 | 5 |
| Malignant_lesions | 18 | 154 | 4 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8930    0.9231    0.9078       208
     Malignant_lesions     0.9333    0.8750    0.9032       176
body_of_stomach_normal     0.9541    0.9740    0.9639       192

              accuracy                         0.9253       576
             macro avg     0.9268    0.9240    0.9250       576
          weighted avg     0.9257    0.9253    0.9251       576

```