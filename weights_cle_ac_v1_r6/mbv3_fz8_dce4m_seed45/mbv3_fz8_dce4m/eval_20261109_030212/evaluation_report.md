# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz8_dce4m_seed45/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9652 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9980 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9722 |
| Balanced Acc | 0.9720 |
| Macro-Sensitivity | 0.9720 |
| Macro-Specificity | 0.9871 |
| Macro-Precision | 0.9594 |
| AUC-PR (Macro) | 0.9941 |
| Cohen's Kappa | 0.9556 |
| MCC | 0.9559 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 2341.5 (0.43 ms/张) |
| End-to-End FPS | 10.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9538 | 0.9871 | 0.9841 | 0.9688 | 0.9622 | 0.9979 | 0.9975 | 195 |
| Malignant_lesions | 0.9620 | 0.9741 | 0.8941 | 0.9268 | 0.9912 | 0.9961 | 0.9849 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=3 TN=229 FN=9
- **Malignant_lesions**: TP=76 FP=9 TN=339 FN=3
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 9 | 0 |
| Malignant_lesions | 3 | 76 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9841    0.9538    0.9688       195
     Malignant_lesions     0.8941    0.9620    0.9268        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9719       427
             macro avg     0.9594    0.9720    0.9652       427
          weighted avg     0.9732    0.9719    0.9722       427

```