# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce4s_seed42/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9630 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9964 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9697 |
| Balanced Acc | 0.9677 |
| Macro-Sensitivity | 0.9677 |
| Macro-Specificity | 0.9854 |
| Macro-Precision | 0.9587 |
| AUC-PR (Macro) | 0.9881 |
| Cohen's Kappa | 0.9518 |
| MCC | 0.9520 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4006.4 (0.25 ms/张) |
| End-to-End FPS | 11.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9538 | 0.9828 | 0.9789 | 0.9662 | 0.9620 | 0.9965 | 0.9961 | 195 |
| Malignant_lesions | 0.9494 | 0.9770 | 0.9036 | 0.9259 | 0.9884 | 0.9928 | 0.9683 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=4 TN=228 FN=9
- **Malignant_lesions**: TP=75 FP=8 TN=340 FN=4
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 8 | 1 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9789    0.9538    0.9662       195
     Malignant_lesions     0.9036    0.9494    0.9259        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9696       427
             macro avg     0.9587    0.9677    0.9630       427
          weighted avg     0.9702    0.9696    0.9697       427

```