# mbv3_r6_s2x_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2x_fz16_seed46/mbv3_r6_s2x_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9395 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9905 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9393 |
| Balanced Acc | 0.9382 |
| Macro-Sensitivity | 0.9382 |
| Macro-Specificity | 0.9691 |
| Macro-Precision | 0.9415 |
| AUC-PR (Macro) | 0.9830 |
| Cohen's Kappa | 0.9085 |
| MCC | 0.9089 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.10M |
| FLOPs | 464.20M |
| Params | 3.39M |
| Pure-Inference FPS | 5568.8 (0.18 ms/张) |
| End-to-End FPS | 293.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9429 | 0.9037 | 0.9249 | 0.9693 | 0.9848 | 0.9743 | 208 |
| Malignant_lesions | 0.9091 | 0.9800 | 0.9524 | 0.9302 | 0.9608 | 0.9900 | 0.9805 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9844 | 0.9684 | 0.9634 | 0.9793 | 0.9969 | 0.9942 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=21 TN=347 FN=11
- **Malignant_lesions**: TP=160 FP=8 TN=392 FN=16
- **body_of_stomach_normal**: TP=184 FP=6 TN=378 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 8 | 3 |
| Malignant_lesions | 13 | 160 | 3 |
| body_of_stomach_normal | 8 | 0 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9037    0.9471    0.9249       208
     Malignant_lesions     0.9524    0.9091    0.9302       176
body_of_stomach_normal     0.9684    0.9583    0.9634       192

              accuracy                         0.9392       576
             macro avg     0.9415    0.9382    0.9395       576
          weighted avg     0.9401    0.9392    0.9393       576

```