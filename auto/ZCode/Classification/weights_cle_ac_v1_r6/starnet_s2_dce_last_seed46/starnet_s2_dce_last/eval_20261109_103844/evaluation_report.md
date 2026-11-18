# starnet_s2_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce_last_seed46/starnet_s2_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9348 |
| Accuracy | 0.9358 |
| Macro-AUROC (OvR) | 0.9930 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9353 |
| Balanced Acc | 0.9329 |
| Macro-Sensitivity | 0.9329 |
| Macro-Specificity | 0.9671 |
| Macro-Precision | 0.9398 |
| AUC-PR (Macro) | 0.9875 |
| Cohen's Kappa | 0.9032 |
| MCC | 0.9046 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4914.9 (0.20 ms/张) |
| End-to-End FPS | 8.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9321 | 0.8884 | 0.9213 | 0.9744 | 0.9899 | 0.9824 | 208 |
| Malignant_lesions | 0.8523 | 0.9850 | 0.9615 | 0.9036 | 0.9381 | 0.9903 | 0.9827 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9844 | 0.9694 | 0.9794 | 0.9947 | 0.9987 | 0.9975 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=25 TN=343 FN=9
- **Malignant_lesions**: TP=150 FP=6 TN=394 FN=26
- **body_of_stomach_normal**: TP=190 FP=6 TN=378 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 6 | 3 |
| Malignant_lesions | 23 | 150 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8884    0.9567    0.9213       208
     Malignant_lesions     0.9615    0.8523    0.9036       176
body_of_stomach_normal     0.9694    0.9896    0.9794       192

              accuracy                         0.9358       576
             macro avg     0.9398    0.9329    0.9348       576
          weighted avg     0.9377    0.9358    0.9353       576

```