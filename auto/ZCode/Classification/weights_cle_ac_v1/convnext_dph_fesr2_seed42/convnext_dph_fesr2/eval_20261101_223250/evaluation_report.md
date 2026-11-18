# convnext_dph_fesr2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_fesr2_seed42/convnext_dph_fesr2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9736 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9986 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9790 |
| Balanced Acc | 0.9771 |
| Macro-Sensitivity | 0.9771 |
| Macro-Specificity | 0.9899 |
| Macro-Precision | 0.9704 |
| AUC-PR (Macro) | 0.9953 |
| Cohen's Kappa | 0.9666 |
| MCC | 0.9667 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 434.9 (2.30 ms/张) |
| End-to-End FPS | 180.7 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9871 | 0.9844 | 0.9767 | 0.9745 | 0.9986 | 0.9983 | 195 |
| Malignant_lesions | 0.9620 | 0.9828 | 0.9268 | 0.9441 | 0.9913 | 0.9971 | 0.9876 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=3 TN=229 FN=6
- **Malignant_lesions**: TP=76 FP=6 TN=342 FN=3
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 6 | 0 |
| Malignant_lesions | 3 | 76 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9844    0.9692    0.9767       195
     Malignant_lesions     0.9268    0.9620    0.9441        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9789       427
             macro avg     0.9704    0.9771    0.9736       427
          weighted avg     0.9793    0.9789    0.9790       427

```