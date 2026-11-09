# convnext_tiny_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_dce4m_seed43/convnext_tiny_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9492 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9495 |
| Balanced Acc | 0.9488 |
| Macro-Sensitivity | 0.9488 |
| Macro-Specificity | 0.9746 |
| Macro-Precision | 0.9500 |
| AUC-PR (Macro) | 0.9921 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9244 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1464.7 (0.68 ms/张) |
| End-to-End FPS | 7.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9620 | 0.9336 | 0.9403 | 0.9699 | 0.9945 | 0.9908 | 208 |
| Malignant_lesions | 0.9148 | 0.9775 | 0.9471 | 0.9306 | 0.9631 | 0.9946 | 0.9874 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9844 | 0.9692 | 0.9767 | 0.9921 | 0.9990 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=14 TN=354 FN=11
- **Malignant_lesions**: TP=161 FP=9 TN=391 FN=15
- **body_of_stomach_normal**: TP=189 FP=6 TN=378 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 11 | 161 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9336    0.9471    0.9403       208
     Malignant_lesions     0.9471    0.9148    0.9306       176
body_of_stomach_normal     0.9692    0.9844    0.9767       192

              accuracy                         0.9497       576
             macro avg     0.9500    0.9488    0.9492       576
          weighted avg     0.9496    0.9497    0.9495       576

```