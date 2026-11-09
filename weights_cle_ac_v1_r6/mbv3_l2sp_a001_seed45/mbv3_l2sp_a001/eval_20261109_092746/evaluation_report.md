# mbv3_l2sp_a001 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_l2sp_a001_seed45/mbv3_l2sp_a001/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9375 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9925 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9375 |
| Balanced Acc | 0.9354 |
| Macro-Sensitivity | 0.9354 |
| Macro-Specificity | 0.9681 |
| Macro-Precision | 0.9414 |
| AUC-PR (Macro) | 0.9858 |
| Cohen's Kappa | 0.9058 |
| MCC | 0.9068 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3705.5 (0.27 ms/张) |
| End-to-End FPS | 123.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9348 | 0.8929 | 0.9259 | 0.9773 | 0.9894 | 0.9830 | 208 |
| Malignant_lesions | 0.8864 | 0.9850 | 0.9630 | 0.9231 | 0.9517 | 0.9898 | 0.9777 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9844 | 0.9684 | 0.9634 | 0.9793 | 0.9983 | 0.9968 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=24 TN=344 FN=8
- **Malignant_lesions**: TP=156 FP=6 TN=394 FN=20
- **body_of_stomach_normal**: TP=184 FP=6 TN=378 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 5 | 3 |
| Malignant_lesions | 17 | 156 | 3 |
| body_of_stomach_normal | 7 | 1 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8929    0.9615    0.9259       208
     Malignant_lesions     0.9630    0.8864    0.9231       176
body_of_stomach_normal     0.9684    0.9583    0.9634       192

              accuracy                         0.9375       576
             macro avg     0.9414    0.9354    0.9375       576
          weighted avg     0.9395    0.9375    0.9375       576

```