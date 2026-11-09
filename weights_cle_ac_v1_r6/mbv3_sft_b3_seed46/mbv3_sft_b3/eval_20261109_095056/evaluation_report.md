# mbv3_sft_b3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b3_seed46/mbv3_sft_b3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9430 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9911 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9429 |
| Balanced Acc | 0.9416 |
| Macro-Sensitivity | 0.9416 |
| Macro-Specificity | 0.9709 |
| Macro-Precision | 0.9453 |
| AUC-PR (Macro) | 0.9836 |
| Cohen's Kappa | 0.9138 |
| MCC | 0.9143 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3877.7 (0.26 ms/张) |
| End-to-End FPS | 94.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9429 | 0.9050 | 0.9324 | 0.9775 | 0.9866 | 0.9767 | 208 |
| Malignant_lesions | 0.9205 | 0.9775 | 0.9474 | 0.9337 | 0.9654 | 0.9900 | 0.9803 | 176 |
| body_of_stomach_normal | 0.9427 | 0.9922 | 0.9837 | 0.9628 | 0.9719 | 0.9966 | 0.9939 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=21 TN=347 FN=8
- **Malignant_lesions**: TP=162 FP=9 TN=391 FN=14
- **body_of_stomach_normal**: TP=181 FP=3 TN=381 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 13 | 162 | 1 |
| body_of_stomach_normal | 8 | 3 | 181 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9050    0.9615    0.9324       208
     Malignant_lesions     0.9474    0.9205    0.9337       176
body_of_stomach_normal     0.9837    0.9427    0.9628       192

              accuracy                         0.9427       576
             macro avg     0.9453    0.9416    0.9430       576
          weighted avg     0.9442    0.9427    0.9429       576

```