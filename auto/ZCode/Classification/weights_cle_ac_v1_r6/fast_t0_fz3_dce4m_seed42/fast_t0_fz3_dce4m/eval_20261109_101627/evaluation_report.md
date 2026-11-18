# fast_t0_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_dce4m_seed42/fast_t0_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9459 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9461 |
| Balanced Acc | 0.9447 |
| Macro-Sensitivity | 0.9447 |
| Macro-Specificity | 0.9727 |
| Macro-Precision | 0.9479 |
| AUC-PR (Macro) | 0.9901 |
| Cohen's Kappa | 0.9190 |
| MCC | 0.9194 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 6449.2 (0.16 ms/张) |
| End-to-End FPS | 8.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9511 | 0.9171 | 0.9365 | 0.9749 | 0.9919 | 0.9874 | 208 |
| Malignant_lesions | 0.9034 | 0.9825 | 0.9578 | 0.9298 | 0.9585 | 0.9940 | 0.9850 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9844 | 0.9689 | 0.9714 | 0.9869 | 0.9989 | 0.9980 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=18 TN=350 FN=9
- **Malignant_lesions**: TP=159 FP=7 TN=393 FN=17
- **body_of_stomach_normal**: TP=187 FP=6 TN=378 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 13 | 159 | 4 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9171    0.9567    0.9365       208
     Malignant_lesions     0.9578    0.9034    0.9298       176
body_of_stomach_normal     0.9689    0.9740    0.9714       192

              accuracy                         0.9462       576
             macro avg     0.9479    0.9447    0.9459       576
          weighted avg     0.9468    0.9462    0.9461       576

```