# fast_t0_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_seed42/fast_t0_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9425 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9932 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9426 |
| Balanced Acc | 0.9419 |
| Macro-Sensitivity | 0.9419 |
| Macro-Specificity | 0.9711 |
| Macro-Precision | 0.9432 |
| AUC-PR (Macro) | 0.9862 |
| Cohen's Kappa | 0.9138 |
| MCC | 0.9139 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 4318.4 (0.23 ms/张) |
| End-to-End FPS | 64.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9538 | 0.9198 | 0.9286 | 0.9643 | 0.9897 | 0.9843 | 208 |
| Malignant_lesions | 0.9091 | 0.9725 | 0.9357 | 0.9222 | 0.9605 | 0.9906 | 0.9756 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9870 | 0.9741 | 0.9766 | 0.9896 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=17 TN=351 FN=13
- **Malignant_lesions**: TP=160 FP=11 TN=389 FN=16
- **body_of_stomach_normal**: TP=188 FP=5 TN=379 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 11 | 2 |
| Malignant_lesions | 13 | 160 | 3 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9198    0.9375    0.9286       208
     Malignant_lesions     0.9357    0.9091    0.9222       176
body_of_stomach_normal     0.9741    0.9792    0.9766       192

              accuracy                         0.9427       576
             macro avg     0.9432    0.9419    0.9425       576
          weighted avg     0.9428    0.9427    0.9426       576

```