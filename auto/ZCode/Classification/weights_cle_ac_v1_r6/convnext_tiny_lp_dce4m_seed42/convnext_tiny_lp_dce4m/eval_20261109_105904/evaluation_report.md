# convnext_tiny_lp_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_lp_dce4m_seed42/convnext_tiny_lp_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9498 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9957 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9495 |
| Balanced Acc | 0.9493 |
| Macro-Sensitivity | 0.9493 |
| Macro-Specificity | 0.9745 |
| Macro-Precision | 0.9509 |
| AUC-PR (Macro) | 0.9921 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9246 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1657.7 (0.60 ms/张) |
| End-to-End FPS | 215.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9674 | 0.9420 | 0.9398 | 0.9648 | 0.9931 | 0.9891 | 208 |
| Malignant_lesions | 0.9261 | 0.9875 | 0.9702 | 0.9477 | 0.9681 | 0.9963 | 0.9918 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9688 | 0.9403 | 0.9618 | 0.9920 | 0.9976 | 0.9953 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=12 TN=356 FN=13
- **Malignant_lesions**: TP=163 FP=5 TN=395 FN=13
- **body_of_stomach_normal**: TP=189 FP=12 TN=372 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 5 | 8 |
| Malignant_lesions | 9 | 163 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9420    0.9375    0.9398       208
     Malignant_lesions     0.9702    0.9261    0.9477       176
body_of_stomach_normal     0.9403    0.9844    0.9618       192

              accuracy                         0.9497       576
             macro avg     0.9509    0.9493    0.9498       576
          weighted avg     0.9501    0.9497    0.9495       576

```