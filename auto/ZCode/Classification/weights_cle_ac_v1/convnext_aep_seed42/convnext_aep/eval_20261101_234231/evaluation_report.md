# convnext_aep 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_aep_seed42/convnext_aep/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9481 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9481 |
| Balanced Acc | 0.9488 |
| Macro-Sensitivity | 0.9488 |
| Macro-Specificity | 0.9741 |
| Macro-Precision | 0.9478 |
| AUC-PR (Macro) | 0.9905 |
| Cohen's Kappa | 0.9218 |
| MCC | 0.9220 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 983.5 (1.02 ms/张) |
| End-to-End FPS | 237.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9701 | 0.9461 | 0.9369 | 0.9597 | 0.9919 | 0.9866 | 208 |
| Malignant_lesions | 0.9602 | 0.9600 | 0.9135 | 0.9363 | 0.9821 | 0.9942 | 0.9873 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9922 | 0.9840 | 0.9710 | 0.9794 | 0.9988 | 0.9976 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=11 TN=357 FN=15
- **Malignant_lesions**: TP=169 FP=16 TN=384 FN=7
- **body_of_stomach_normal**: TP=184 FP=3 TN=381 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 14 | 1 |
| Malignant_lesions | 5 | 169 | 2 |
| body_of_stomach_normal | 6 | 2 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9461    0.9279    0.9369       208
     Malignant_lesions     0.9135    0.9602    0.9363       176
body_of_stomach_normal     0.9840    0.9583    0.9710       192

              accuracy                         0.9479       576
             macro avg     0.9478    0.9488    0.9481       576
          weighted avg     0.9488    0.9479    0.9481       576

```