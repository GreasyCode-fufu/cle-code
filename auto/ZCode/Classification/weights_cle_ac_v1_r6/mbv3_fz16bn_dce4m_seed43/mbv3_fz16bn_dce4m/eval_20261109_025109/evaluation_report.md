# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16bn_dce4m_seed43/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9356 |
| Accuracy | 0.9358 |
| Macro-AUROC (OvR) | 0.9926 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9357 |
| Balanced Acc | 0.9344 |
| Macro-Sensitivity | 0.9344 |
| Macro-Specificity | 0.9674 |
| Macro-Precision | 0.9378 |
| AUC-PR (Macro) | 0.9863 |
| Cohen's Kappa | 0.9033 |
| MCC | 0.9037 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3349.3 (0.30 ms/张) |
| End-to-End FPS | 14.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9429 | 0.9032 | 0.9224 | 0.9666 | 0.9884 | 0.9810 | 208 |
| Malignant_lesions | 0.8920 | 0.9800 | 0.9515 | 0.9208 | 0.9538 | 0.9918 | 0.9819 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9792 | 0.9588 | 0.9637 | 0.9843 | 0.9977 | 0.9959 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=21 TN=347 FN=12
- **Malignant_lesions**: TP=157 FP=8 TN=392 FN=19
- **body_of_stomach_normal**: TP=186 FP=8 TN=376 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 8 | 4 |
| Malignant_lesions | 15 | 157 | 4 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9032    0.9423    0.9224       208
     Malignant_lesions     0.9515    0.8920    0.9208       176
body_of_stomach_normal     0.9588    0.9688    0.9637       192

              accuracy                         0.9358       576
             macro avg     0.9378    0.9344    0.9356       576
          weighted avg     0.9365    0.9358    0.9357       576

```