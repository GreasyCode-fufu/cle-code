# starnet_s2_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce4m_seed45/starnet_s2_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9689 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9970 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9761 |
| Balanced Acc | 0.9603 |
| Macro-Sensitivity | 0.9603 |
| Macro-Specificity | 0.9866 |
| Macro-Precision | 0.9794 |
| AUC-PR (Macro) | 0.9919 |
| Cohen's Kappa | 0.9625 |
| MCC | 0.9630 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3621.5 (0.28 ms/张) |
| End-to-End FPS | 8.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9949 | 0.9698 | 0.9652 | 0.9798 | 0.9956 | 0.9971 | 0.9964 | 195 |
| Malignant_lesions | 0.8861 | 0.9971 | 0.9859 | 0.9333 | 0.9747 | 0.9941 | 0.9795 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 0.9999 | 0.9999 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=7 TN=225 FN=1
- **Malignant_lesions**: TP=70 FP=1 TN=347 FN=9
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 1 | 0 |
| Malignant_lesions | 7 | 70 | 2 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9652    0.9949    0.9798       195
     Malignant_lesions     0.9859    0.8861    0.9333        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9766       427
             macro avg     0.9794    0.9603    0.9689       427
          weighted avg     0.9769    0.9766    0.9761       427

```