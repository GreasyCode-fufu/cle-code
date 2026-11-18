# starnet_s2_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce_last_seed44/starnet_s2_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9373 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9904 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9373 |
| Balanced Acc | 0.9373 |
| Macro-Sensitivity | 0.9373 |
| Macro-Specificity | 0.9686 |
| Macro-Precision | 0.9376 |
| AUC-PR (Macro) | 0.9819 |
| Cohen's Kappa | 0.9060 |
| MCC | 0.9061 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 5317.7 (0.19 ms/张) |
| End-to-End FPS | 7.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9592 | 0.9275 | 0.9253 | 0.9566 | 0.9861 | 0.9777 | 208 |
| Malignant_lesions | 0.9148 | 0.9725 | 0.9360 | 0.9253 | 0.9629 | 0.9871 | 0.9720 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9740 | 0.9492 | 0.9614 | 0.9868 | 0.9980 | 0.9961 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=15 TN=353 FN=16
- **Malignant_lesions**: TP=161 FP=11 TN=389 FN=15
- **body_of_stomach_normal**: TP=187 FP=10 TN=374 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 11 | 5 |
| Malignant_lesions | 10 | 161 | 5 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9275    0.9231    0.9253       208
     Malignant_lesions     0.9360    0.9148    0.9253       176
body_of_stomach_normal     0.9492    0.9740    0.9614       192

              accuracy                         0.9375       576
             macro avg     0.9376    0.9373    0.9373       576
          weighted avg     0.9374    0.9375    0.9373       576

```