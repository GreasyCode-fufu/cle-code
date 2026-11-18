# mbv3_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce16_seed45/mbv3_dce16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9426 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9901 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9428 |
| Balanced Acc | 0.9418 |
| Macro-Sensitivity | 0.9418 |
| Macro-Specificity | 0.9711 |
| Macro-Precision | 0.9437 |
| AUC-PR (Macro) | 0.9810 |
| Cohen's Kappa | 0.9138 |
| MCC | 0.9139 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 5678.1 (0.18 ms/张) |
| End-to-End FPS | 16.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9511 | 0.9163 | 0.9314 | 0.9695 | 0.9863 | 0.9762 | 208 |
| Malignant_lesions | 0.9148 | 0.9725 | 0.9360 | 0.9253 | 0.9629 | 0.9862 | 0.9708 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9896 | 0.9788 | 0.9711 | 0.9819 | 0.9979 | 0.9961 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=18 TN=350 FN=11
- **Malignant_lesions**: TP=161 FP=11 TN=389 FN=15
- **body_of_stomach_normal**: TP=185 FP=4 TN=380 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 13 | 161 | 2 |
| body_of_stomach_normal | 5 | 2 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9163    0.9471    0.9314       208
     Malignant_lesions     0.9360    0.9148    0.9253       176
body_of_stomach_normal     0.9788    0.9635    0.9711       192

              accuracy                         0.9427       576
             macro avg     0.9437    0.9418    0.9426       576
          weighted avg     0.9432    0.9427    0.9428       576

```