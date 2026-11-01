# convnext_fesr2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fesr2_seed42/convnext_fesr2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9723 |
| Accuracy | 0.9722 |
| Macro-AUROC (OvR) | 0.9970 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9723 |
| Balanced Acc | 0.9718 |
| Macro-Sensitivity | 0.9718 |
| Macro-Specificity | 0.9859 |
| Macro-Precision | 0.9729 |
| AUC-PR (Macro) | 0.9944 |
| Cohen's Kappa | 0.9582 |
| MCC | 0.9583 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 485.3 (2.06 ms/张) |
| End-to-End FPS | 188.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9728 | 0.9528 | 0.9619 | 0.9835 | 0.9957 | 0.9925 | 208 |
| Malignant_lesions | 0.9545 | 0.9875 | 0.9711 | 0.9628 | 0.9801 | 0.9955 | 0.9914 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9974 | 0.9948 | 0.9922 | 0.9948 | 0.9997 | 0.9994 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=10 TN=358 FN=6
- **Malignant_lesions**: TP=168 FP=5 TN=395 FN=8
- **body_of_stomach_normal**: TP=190 FP=1 TN=383 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 8 | 168 | 0 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9528    0.9712    0.9619       208
     Malignant_lesions     0.9711    0.9545    0.9628       176
body_of_stomach_normal     0.9948    0.9896    0.9922       192

              accuracy                         0.9722       576
             macro avg     0.9729    0.9718    0.9723       576
          weighted avg     0.9724    0.9722    0.9723       576

```