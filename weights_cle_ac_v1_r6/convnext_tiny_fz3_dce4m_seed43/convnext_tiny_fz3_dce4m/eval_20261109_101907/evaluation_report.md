# convnext_tiny_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_dce4m_seed43/convnext_tiny_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9789 |
| Accuracy | 0.9836 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9836 |
| Balanced Acc | 0.9780 |
| Macro-Sensitivity | 0.9780 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9799 |
| AUC-PR (Macro) | 0.9977 |
| Cohen's Kappa | 0.9739 |
| MCC | 0.9739 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1471.8 (0.68 ms/张) |
| End-to-End FPS | 7.7 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9871 | 0.9846 | 0.9846 | 0.9871 | 0.9994 | 0.9993 | 195 |
| Malignant_lesions | 0.9494 | 0.9914 | 0.9615 | 0.9554 | 0.9885 | 0.9984 | 0.9937 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=3 TN=229 FN=3
- **Malignant_lesions**: TP=75 FP=3 TN=345 FN=4
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 3 | 0 |
| Malignant_lesions | 3 | 75 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9846    0.9846    0.9846       195
     Malignant_lesions     0.9615    0.9494    0.9554        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9836       427
             macro avg     0.9799    0.9780    0.9789       427
          weighted avg     0.9835    0.9836    0.9836       427

```