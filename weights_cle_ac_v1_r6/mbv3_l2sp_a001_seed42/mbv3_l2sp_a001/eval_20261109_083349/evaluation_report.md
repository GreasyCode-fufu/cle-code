# mbv3_l2sp_a001 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_l2sp_a001_seed42/mbv3_l2sp_a001/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9429 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9931 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9428 |
| Balanced Acc | 0.9409 |
| Macro-Sensitivity | 0.9409 |
| Macro-Specificity | 0.9707 |
| Macro-Precision | 0.9466 |
| AUC-PR (Macro) | 0.9870 |
| Cohen's Kappa | 0.9137 |
| MCC | 0.9145 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3496.3 (0.29 ms/张) |
| End-to-End FPS | 33.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9402 | 0.9009 | 0.9302 | 0.9774 | 0.9888 | 0.9808 | 208 |
| Malignant_lesions | 0.8977 | 0.9900 | 0.9753 | 0.9349 | 0.9565 | 0.9925 | 0.9841 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9818 | 0.9635 | 0.9635 | 0.9818 | 0.9979 | 0.9961 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=22 TN=346 FN=8
- **Malignant_lesions**: TP=158 FP=4 TN=396 FN=18
- **body_of_stomach_normal**: TP=185 FP=7 TN=377 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 4 | 4 |
| Malignant_lesions | 15 | 158 | 3 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9009    0.9615    0.9302       208
     Malignant_lesions     0.9753    0.8977    0.9349       176
body_of_stomach_normal     0.9635    0.9635    0.9635       192

              accuracy                         0.9427       576
             macro avg     0.9466    0.9409    0.9429       576
          weighted avg     0.9445    0.9427    0.9428       576

```