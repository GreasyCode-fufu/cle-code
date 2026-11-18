# mbv3_sft_b1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b1_seed43/mbv3_sft_b1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8882 |
| Accuracy | 0.8872 |
| Macro-AUROC (OvR) | 0.9810 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8882 |
| Balanced Acc | 0.8839 |
| Macro-Sensitivity | 0.8839 |
| Macro-Specificity | 0.9423 |
| Macro-Precision | 0.8983 |
| AUC-PR (Macro) | 0.9631 |
| Cohen's Kappa | 0.8297 |
| MCC | 0.8328 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3899.7 (0.26 ms/张) |
| End-to-End FPS | 110.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.8696 | 0.8008 | 0.8597 | 0.9552 | 0.9695 | 0.9544 | 208 |
| Malignant_lesions | 0.8125 | 0.9650 | 0.9108 | 0.8589 | 0.9212 | 0.9765 | 0.9407 | 176 |
| body_of_stomach_normal | 0.9115 | 0.9922 | 0.9831 | 0.9459 | 0.9573 | 0.9971 | 0.9943 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=48 TN=320 FN=15
- **Malignant_lesions**: TP=143 FP=14 TN=386 FN=33
- **body_of_stomach_normal**: TP=175 FP=3 TN=381 FN=17

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 12 | 3 |
| Malignant_lesions | 33 | 143 | 0 |
| body_of_stomach_normal | 15 | 2 | 175 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8008    0.9279    0.8597       208
     Malignant_lesions     0.9108    0.8125    0.8589       176
body_of_stomach_normal     0.9831    0.9115    0.9459       192

              accuracy                         0.8872       576
             macro avg     0.8983    0.8839    0.8882       576
          weighted avg     0.8952    0.8872    0.8882       576

```