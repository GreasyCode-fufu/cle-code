# convnext_tiny_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_dce_last_seed45/convnext_tiny_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9457 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9943 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9459 |
| Balanced Acc | 0.9450 |
| Macro-Sensitivity | 0.9450 |
| Macro-Specificity | 0.9728 |
| Macro-Precision | 0.9472 |
| AUC-PR (Macro) | 0.9889 |
| Cohen's Kappa | 0.9190 |
| MCC | 0.9193 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1532.7 (0.65 ms/张) |
| End-to-End FPS | 22.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9592 | 0.9292 | 0.9381 | 0.9698 | 0.9931 | 0.9888 | 208 |
| Malignant_lesions | 0.9034 | 0.9825 | 0.9578 | 0.9298 | 0.9585 | 0.9921 | 0.9827 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9766 | 0.9545 | 0.9692 | 0.9921 | 0.9976 | 0.9952 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=15 TN=353 FN=11
- **Malignant_lesions**: TP=159 FP=7 TN=393 FN=17
- **body_of_stomach_normal**: TP=189 FP=9 TN=375 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 7 | 4 |
| Malignant_lesions | 12 | 159 | 5 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9292    0.9471    0.9381       208
     Malignant_lesions     0.9578    0.9034    0.9298       176
body_of_stomach_normal     0.9545    0.9844    0.9692       192

              accuracy                         0.9462       576
             macro avg     0.9472    0.9450    0.9457       576
          weighted avg     0.9464    0.9462    0.9459       576

```