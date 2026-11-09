# fast_t0_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_dce_last_seed44/fast_t0_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9335 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9885 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9339 |
| Balanced Acc | 0.9314 |
| Macro-Sensitivity | 0.9314 |
| Macro-Specificity | 0.9663 |
| Macro-Precision | 0.9378 |
| AUC-PR (Macro) | 0.9778 |
| Cohen's Kappa | 0.9006 |
| MCC | 0.9017 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 6106.5 (0.16 ms/张) |
| End-to-End FPS | 8.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9293 | 0.8844 | 0.9192 | 0.9744 | 0.9842 | 0.9743 | 208 |
| Malignant_lesions | 0.8636 | 0.9800 | 0.9500 | 0.9048 | 0.9423 | 0.9826 | 0.9617 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9896 | 0.9791 | 0.9765 | 0.9870 | 0.9986 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=26 TN=342 FN=9
- **Malignant_lesions**: TP=152 FP=8 TN=392 FN=24
- **body_of_stomach_normal**: TP=187 FP=4 TN=380 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 21 | 152 | 3 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8844    0.9567    0.9192       208
     Malignant_lesions     0.9500    0.8636    0.9048       176
body_of_stomach_normal     0.9791    0.9740    0.9765       192

              accuracy                         0.9340       576
             macro avg     0.9378    0.9314    0.9335       576
          weighted avg     0.9360    0.9340    0.9339       576

```