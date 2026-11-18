# convnext_dph_am2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_am2_seed42/convnext_dph_am2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9577 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9582 |
| Balanced Acc | 0.9573 |
| Macro-Sensitivity | 0.9573 |
| Macro-Specificity | 0.9790 |
| Macro-Precision | 0.9583 |
| AUC-PR (Macro) | 0.9925 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9374 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 671.0 (1.49 ms/张) |
| End-to-End FPS | 212.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9674 | 0.9431 | 0.9499 | 0.9753 | 0.9946 | 0.9908 | 208 |
| Malignant_lesions | 0.9205 | 0.9775 | 0.9474 | 0.9337 | 0.9654 | 0.9942 | 0.9873 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9922 | 0.9845 | 0.9896 | 0.9974 | 0.9996 | 0.9994 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=12 TN=356 FN=9
- **Malignant_lesions**: TP=162 FP=9 TN=391 FN=14
- **body_of_stomach_normal**: TP=191 FP=3 TN=381 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 9 | 0 |
| Malignant_lesions | 11 | 162 | 3 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9431    0.9567    0.9499       208
     Malignant_lesions     0.9474    0.9205    0.9337       176
body_of_stomach_normal     0.9845    0.9948    0.9896       192

              accuracy                         0.9583       576
             macro avg     0.9583    0.9573    0.9577       576
          weighted avg     0.9582    0.9583    0.9582       576

```