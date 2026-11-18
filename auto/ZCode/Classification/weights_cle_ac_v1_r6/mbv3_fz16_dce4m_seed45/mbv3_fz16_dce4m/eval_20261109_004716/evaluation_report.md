# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce4m_seed45/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9655 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9654 |
| Balanced Acc | 0.9657 |
| Macro-Sensitivity | 0.9657 |
| Macro-Specificity | 0.9826 |
| Macro-Precision | 0.9655 |
| AUC-PR (Macro) | 0.9904 |
| Cohen's Kappa | 0.9478 |
| MCC | 0.9478 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5631.8 (0.18 ms/张) |
| End-to-End FPS | 226.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9728 | 0.9519 | 0.9519 | 0.9728 | 0.9918 | 0.9881 | 208 |
| Malignant_lesions | 0.9659 | 0.9775 | 0.9497 | 0.9577 | 0.9849 | 0.9935 | 0.9837 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9974 | 0.9947 | 0.9869 | 0.9897 | 0.9997 | 0.9995 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=10 TN=358 FN=10
- **Malignant_lesions**: TP=170 FP=9 TN=391 FN=6
- **body_of_stomach_normal**: TP=188 FP=1 TN=383 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 6 | 170 | 0 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9519    0.9519    0.9519       208
     Malignant_lesions     0.9497    0.9659    0.9577       176
body_of_stomach_normal     0.9947    0.9792    0.9869       192

              accuracy                         0.9653       576
             macro avg     0.9655    0.9657    0.9655       576
          weighted avg     0.9655    0.9653    0.9654       576

```