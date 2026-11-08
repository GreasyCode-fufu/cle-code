# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce4m_seed44/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9624 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9971 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9699 |
| Balanced Acc | 0.9702 |
| Macro-Sensitivity | 0.9702 |
| Macro-Specificity | 0.9861 |
| Macro-Precision | 0.9559 |
| AUC-PR (Macro) | 0.9900 |
| Cohen's Kappa | 0.9519 |
| MCC | 0.9524 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4063.1 (0.25 ms/张) |
| End-to-End FPS | 216.9 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9487 | 0.9871 | 0.9840 | 0.9661 | 0.9582 | 0.9971 | 0.9968 | 195 |
| Malignant_lesions | 0.9620 | 0.9713 | 0.8837 | 0.9212 | 0.9912 | 0.9942 | 0.9731 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=185 FP=3 TN=229 FN=10
- **Malignant_lesions**: TP=76 FP=10 TN=338 FN=3
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 185 | 10 | 0 |
| Malignant_lesions | 3 | 76 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9840    0.9487    0.9661       195
     Malignant_lesions     0.8837    0.9620    0.9212        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9696       427
             macro avg     0.9559    0.9702    0.9624       427
          weighted avg     0.9712    0.9696    0.9699       427

```