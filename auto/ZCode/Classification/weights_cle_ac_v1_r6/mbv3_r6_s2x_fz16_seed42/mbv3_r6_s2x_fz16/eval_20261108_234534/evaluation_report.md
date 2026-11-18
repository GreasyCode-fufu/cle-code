# mbv3_r6_s2x_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2x_fz16_seed42/mbv3_r6_s2x_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9389 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9938 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9391 |
| Balanced Acc | 0.9367 |
| Macro-Sensitivity | 0.9367 |
| Macro-Specificity | 0.9689 |
| Macro-Precision | 0.9438 |
| AUC-PR (Macro) | 0.9886 |
| Cohen's Kappa | 0.9084 |
| MCC | 0.9096 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.10M |
| FLOPs | 464.20M |
| Params | 3.39M |
| Pure-Inference FPS | 4455.0 (0.22 ms/张) |
| End-to-End FPS | 266.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9402 | 0.9018 | 0.9352 | 0.9830 | 0.9904 | 0.9863 | 208 |
| Malignant_lesions | 0.8807 | 0.9925 | 0.9810 | 0.9281 | 0.9498 | 0.9936 | 0.9847 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9740 | 0.9485 | 0.9534 | 0.9791 | 0.9974 | 0.9948 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=22 TN=346 FN=6
- **Malignant_lesions**: TP=155 FP=3 TN=397 FN=21
- **body_of_stomach_normal**: TP=184 FP=10 TN=374 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 3 | 3 |
| Malignant_lesions | 14 | 155 | 7 |
| body_of_stomach_normal | 8 | 0 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9018    0.9712    0.9352       208
     Malignant_lesions     0.9810    0.8807    0.9281       176
body_of_stomach_normal     0.9485    0.9583    0.9534       192

              accuracy                         0.9392       576
             macro avg     0.9438    0.9367    0.9389       576
          weighted avg     0.9415    0.9392    0.9391       576

```