# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mobilenet_v3_large_seed45/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9390 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9900 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9392 |
| Balanced Acc | 0.9372 |
| Macro-Sensitivity | 0.9372 |
| Macro-Specificity | 0.9690 |
| Macro-Precision | 0.9423 |
| AUC-PR (Macro) | 0.9813 |
| Cohen's Kappa | 0.9085 |
| MCC | 0.9092 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 5062.2 (0.20 ms/张) |
| End-to-End FPS | 285.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9429 | 0.9058 | 0.9374 | 0.9830 | 0.9875 | 0.9794 | 208 |
| Malignant_lesions | 0.8977 | 0.9850 | 0.9634 | 0.9294 | 0.9563 | 0.9877 | 0.9731 | 176 |
| body_of_stomach_normal | 0.9427 | 0.9792 | 0.9577 | 0.9501 | 0.9716 | 0.9949 | 0.9915 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=21 TN=347 FN=6
- **Malignant_lesions**: TP=158 FP=6 TN=394 FN=18
- **body_of_stomach_normal**: TP=181 FP=8 TN=376 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 12 | 158 | 6 |
| body_of_stomach_normal | 9 | 2 | 181 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9058    0.9712    0.9374       208
     Malignant_lesions     0.9634    0.8977    0.9294       176
body_of_stomach_normal     0.9577    0.9427    0.9501       192

              accuracy                         0.9392       576
             macro avg     0.9423    0.9372    0.9390       576
          weighted avg     0.9407    0.9392    0.9392       576

```