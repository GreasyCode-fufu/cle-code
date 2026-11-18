# starnet_s2_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce4m_seed45/starnet_s2_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9243 |
| Accuracy | 0.9253 |
| Macro-AUROC (OvR) | 0.9911 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9247 |
| Balanced Acc | 0.9225 |
| Macro-Sensitivity | 0.9225 |
| Macro-Specificity | 0.9619 |
| Macro-Precision | 0.9296 |
| AUC-PR (Macro) | 0.9833 |
| Cohen's Kappa | 0.8875 |
| MCC | 0.8890 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4335.6 (0.23 ms/张) |
| End-to-End FPS | 7.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9293 | 0.8829 | 0.9116 | 0.9661 | 0.9853 | 0.9775 | 208 |
| Malignant_lesions | 0.8409 | 0.9850 | 0.9610 | 0.8970 | 0.9336 | 0.9892 | 0.9749 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9714 | 0.9450 | 0.9643 | 0.9920 | 0.9987 | 0.9974 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=26 TN=342 FN=12
- **Malignant_lesions**: TP=148 FP=6 TN=394 FN=28
- **body_of_stomach_normal**: TP=189 FP=11 TN=373 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 6 | 6 |
| Malignant_lesions | 23 | 148 | 5 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8829    0.9423    0.9116       208
     Malignant_lesions     0.9610    0.8409    0.8970       176
body_of_stomach_normal     0.9450    0.9844    0.9643       192

              accuracy                         0.9253       576
             macro avg     0.9296    0.9225    0.9243       576
          weighted avg     0.9275    0.9253    0.9247       576

```