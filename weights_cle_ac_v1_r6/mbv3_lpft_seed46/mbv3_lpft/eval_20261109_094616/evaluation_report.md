# mbv3_lpft 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_lpft_seed46/mbv3_lpft/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8955 |
| Accuracy | 0.8941 |
| Macro-AUROC (OvR) | 0.9792 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8952 |
| Balanced Acc | 0.8912 |
| Macro-Sensitivity | 0.8912 |
| Macro-Specificity | 0.9457 |
| Macro-Precision | 0.9077 |
| AUC-PR (Macro) | 0.9665 |
| Cohen's Kappa | 0.8402 |
| MCC | 0.8445 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 4613.5 (0.22 ms/张) |
| End-to-End FPS | 203.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.8723 | 0.8089 | 0.8767 | 0.9727 | 0.9767 | 0.9618 | 208 |
| Malignant_lesions | 0.8523 | 0.9675 | 0.9202 | 0.8850 | 0.9370 | 0.9768 | 0.9566 | 176 |
| body_of_stomach_normal | 0.8646 | 0.9974 | 0.9940 | 0.9248 | 0.9364 | 0.9841 | 0.9812 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=47 TN=321 FN=9
- **Malignant_lesions**: TP=150 FP=13 TN=387 FN=26
- **body_of_stomach_normal**: TP=166 FP=1 TN=383 FN=26

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 26 | 150 | 0 |
| body_of_stomach_normal | 21 | 5 | 166 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8089    0.9567    0.8767       208
     Malignant_lesions     0.9202    0.8523    0.8850       176
body_of_stomach_normal     0.9940    0.8646    0.9248       192

              accuracy                         0.8941       576
             macro avg     0.9077    0.8912    0.8955       576
          weighted avg     0.9046    0.8941    0.8952       576

```