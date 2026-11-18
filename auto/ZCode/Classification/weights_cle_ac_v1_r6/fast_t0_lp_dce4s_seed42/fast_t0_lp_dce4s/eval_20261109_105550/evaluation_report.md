# fast_t0_lp_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_lp_dce4s_seed42/fast_t0_lp_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9339 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9925 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9339 |
| Balanced Acc | 0.9328 |
| Macro-Sensitivity | 0.9328 |
| Macro-Specificity | 0.9665 |
| Macro-Precision | 0.9362 |
| AUC-PR (Macro) | 0.9859 |
| Cohen's Kappa | 0.9007 |
| MCC | 0.9011 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 7894.0 (0.13 ms/张) |
| End-to-End FPS | 7.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9457 | 0.9070 | 0.9220 | 0.9640 | 0.9880 | 0.9817 | 208 |
| Malignant_lesions | 0.8920 | 0.9825 | 0.9573 | 0.9235 | 0.9539 | 0.9926 | 0.9819 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9714 | 0.9442 | 0.9563 | 0.9842 | 0.9969 | 0.9942 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=20 TN=348 FN=13
- **Malignant_lesions**: TP=157 FP=7 TN=393 FN=19
- **body_of_stomach_normal**: TP=186 FP=11 TN=373 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 7 | 6 |
| Malignant_lesions | 14 | 157 | 5 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9070    0.9375    0.9220       208
     Malignant_lesions     0.9573    0.8920    0.9235       176
body_of_stomach_normal     0.9442    0.9688    0.9563       192

              accuracy                         0.9340       576
             macro avg     0.9362    0.9328    0.9339       576
          weighted avg     0.9348    0.9340    0.9339       576

```