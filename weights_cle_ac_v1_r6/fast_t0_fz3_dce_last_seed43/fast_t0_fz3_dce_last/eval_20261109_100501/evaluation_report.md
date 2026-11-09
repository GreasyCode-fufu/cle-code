# fast_t0_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_dce_last_seed43/fast_t0_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9551 |
| Accuracy | 0.9649 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9648 |
| Balanced Acc | 0.9543 |
| Macro-Sensitivity | 0.9543 |
| Macro-Specificity | 0.9820 |
| Macro-Precision | 0.9560 |
| AUC-PR (Macro) | 0.9896 |
| Cohen's Kappa | 0.9441 |
| MCC | 0.9441 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 3599.1 (0.28 ms/张) |
| End-to-End FPS | 7.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9641 | 0.9698 | 0.9641 | 0.9641 | 0.9698 | 0.9963 | 0.9956 | 195 |
| Malignant_lesions | 0.8987 | 0.9799 | 0.9103 | 0.9045 | 0.9771 | 0.9933 | 0.9733 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=7 TN=225 FN=7
- **Malignant_lesions**: TP=71 FP=7 TN=341 FN=8
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 7 | 0 |
| Malignant_lesions | 7 | 71 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9641    0.9641    0.9641       195
     Malignant_lesions     0.9103    0.8987    0.9045        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9649       427
             macro avg     0.9560    0.9543    0.9551       427
          weighted avg     0.9647    0.9649    0.9648       427

```