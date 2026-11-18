# convnext_dph_fesr2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_fesr2_seed43/convnext_dph_fesr2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9408 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9951 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9411 |
| Balanced Acc | 0.9424 |
| Macro-Sensitivity | 0.9424 |
| Macro-Specificity | 0.9709 |
| Macro-Precision | 0.9406 |
| AUC-PR (Macro) | 0.9908 |
| Cohen's Kappa | 0.9114 |
| MCC | 0.9121 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 481.1 (2.08 ms/张) |
| End-to-End FPS | 191.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8990 | 0.9783 | 0.9590 | 0.9280 | 0.9449 | 0.9932 | 0.9888 | 208 |
| Malignant_lesions | 0.9489 | 0.9450 | 0.8836 | 0.9151 | 0.9767 | 0.9928 | 0.9849 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9896 | 0.9792 | 0.9792 | 0.9896 | 0.9992 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=8 TN=360 FN=21
- **Malignant_lesions**: TP=167 FP=22 TN=378 FN=9
- **body_of_stomach_normal**: TP=188 FP=4 TN=380 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 19 | 2 |
| Malignant_lesions | 7 | 167 | 2 |
| body_of_stomach_normal | 1 | 3 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9590    0.8990    0.9280       208
     Malignant_lesions     0.8836    0.9489    0.9151       176
body_of_stomach_normal     0.9792    0.9792    0.9792       192

              accuracy                         0.9410       576
             macro avg     0.9406    0.9424    0.9408       576
          weighted avg     0.9427    0.9410    0.9411       576

```