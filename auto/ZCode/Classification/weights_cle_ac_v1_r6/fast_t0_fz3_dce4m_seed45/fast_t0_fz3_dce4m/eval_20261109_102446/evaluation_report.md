# fast_t0_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_dce4m_seed45/fast_t0_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9376 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9922 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9376 |
| Balanced Acc | 0.9366 |
| Macro-Sensitivity | 0.9366 |
| Macro-Specificity | 0.9683 |
| Macro-Precision | 0.9390 |
| AUC-PR (Macro) | 0.9845 |
| Cohen's Kappa | 0.9059 |
| MCC | 0.9062 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 8037.9 (0.12 ms/张) |
| End-to-End FPS | 7.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9429 | 0.9028 | 0.9198 | 0.9639 | 0.9879 | 0.9819 | 208 |
| Malignant_lesions | 0.9034 | 0.9725 | 0.9353 | 0.9191 | 0.9581 | 0.9895 | 0.9736 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9896 | 0.9789 | 0.9738 | 0.9845 | 0.9991 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=21 TN=347 FN=13
- **Malignant_lesions**: TP=159 FP=11 TN=389 FN=17
- **body_of_stomach_normal**: TP=186 FP=4 TN=380 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 11 | 2 |
| Malignant_lesions | 15 | 159 | 2 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9028    0.9375    0.9198       208
     Malignant_lesions     0.9353    0.9034    0.9191       176
body_of_stomach_normal     0.9789    0.9688    0.9738       192

              accuracy                         0.9375       576
             macro avg     0.9390    0.9366    0.9376       576
          weighted avg     0.9381    0.9375    0.9376       576

```