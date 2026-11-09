# fast_t0_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_dce_last_seed42/fast_t0_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9337 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9921 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9339 |
| Balanced Acc | 0.9328 |
| Macro-Sensitivity | 0.9328 |
| Macro-Specificity | 0.9666 |
| Macro-Precision | 0.9353 |
| AUC-PR (Macro) | 0.9844 |
| Cohen's Kappa | 0.9007 |
| MCC | 0.9010 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 8371.7 (0.12 ms/张) |
| End-to-End FPS | 6.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9457 | 0.9070 | 0.9220 | 0.9640 | 0.9874 | 0.9810 | 208 |
| Malignant_lesions | 0.8920 | 0.9750 | 0.9401 | 0.9155 | 0.9535 | 0.9902 | 0.9747 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9792 | 0.9588 | 0.9637 | 0.9843 | 0.9986 | 0.9975 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=20 TN=348 FN=13
- **Malignant_lesions**: TP=157 FP=10 TN=390 FN=19
- **body_of_stomach_normal**: TP=186 FP=8 TN=376 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 10 | 3 |
| Malignant_lesions | 14 | 157 | 5 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9070    0.9375    0.9220       208
     Malignant_lesions     0.9401    0.8920    0.9155       176
body_of_stomach_normal     0.9588    0.9688    0.9637       192

              accuracy                         0.9340       576
             macro avg     0.9353    0.9328    0.9337       576
          weighted avg     0.9344    0.9340    0.9339       576

```