# starnet_s2_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce_last_seed42/starnet_s2_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9378 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9921 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9377 |
| Balanced Acc | 0.9370 |
| Macro-Sensitivity | 0.9370 |
| Macro-Specificity | 0.9684 |
| Macro-Precision | 0.9389 |
| AUC-PR (Macro) | 0.9849 |
| Cohen's Kappa | 0.9060 |
| MCC | 0.9061 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3415.9 (0.29 ms/张) |
| End-to-End FPS | 8.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9457 | 0.9070 | 0.9220 | 0.9640 | 0.9883 | 0.9812 | 208 |
| Malignant_lesions | 0.9205 | 0.9725 | 0.9364 | 0.9284 | 0.9653 | 0.9910 | 0.9785 | 176 |
| body_of_stomach_normal | 0.9531 | 0.9870 | 0.9734 | 0.9632 | 0.9768 | 0.9970 | 0.9949 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=20 TN=348 FN=13
- **Malignant_lesions**: TP=162 FP=11 TN=389 FN=14
- **body_of_stomach_normal**: TP=183 FP=5 TN=379 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 11 | 2 |
| Malignant_lesions | 11 | 162 | 3 |
| body_of_stomach_normal | 9 | 0 | 183 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9070    0.9375    0.9220       208
     Malignant_lesions     0.9364    0.9205    0.9284       176
body_of_stomach_normal     0.9734    0.9531    0.9632       192

              accuracy                         0.9375       576
             macro avg     0.9389    0.9370    0.9378       576
          weighted avg     0.9381    0.9375    0.9377       576

```