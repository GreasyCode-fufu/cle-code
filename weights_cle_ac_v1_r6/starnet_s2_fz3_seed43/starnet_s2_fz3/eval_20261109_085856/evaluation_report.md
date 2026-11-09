# starnet_s2_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_seed43/starnet_s2_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9731 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9987 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9786 |
| Balanced Acc | 0.9670 |
| Macro-Sensitivity | 0.9670 |
| Macro-Specificity | 0.9884 |
| Macro-Precision | 0.9802 |
| AUC-PR (Macro) | 0.9962 |
| Cohen's Kappa | 0.9663 |
| MCC | 0.9666 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 2195.8 (0.46 ms/张) |
| End-to-End FPS | 29.9 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9897 | 0.9828 | 0.9797 | 0.9847 | 0.9913 | 0.9988 | 0.9986 | 195 |
| Malignant_lesions | 0.9114 | 0.9971 | 0.9863 | 0.9474 | 0.9802 | 0.9973 | 0.9901 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9854 | 0.9745 | 0.9871 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=4 TN=228 FN=2
- **Malignant_lesions**: TP=72 FP=1 TN=347 FN=7
- **body_of_stomach_normal**: TP=153 FP=4 TN=270 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 1 | 1 |
| Malignant_lesions | 4 | 72 | 3 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9797    0.9897    0.9847       195
     Malignant_lesions     0.9863    0.9114    0.9474        79
body_of_stomach_normal     0.9745    1.0000    0.9871       153

              accuracy                         0.9789       427
             macro avg     0.9802    0.9670    0.9731       427
          weighted avg     0.9791    0.9789    0.9786       427

```