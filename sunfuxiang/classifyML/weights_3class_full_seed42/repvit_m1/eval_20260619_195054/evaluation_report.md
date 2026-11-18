# repvit_m1 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_full_seed42/repvit_m1/repvit_m1_3class_full_seed42_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7835 |
| Accuracy | 0.7794 |
| Macro-AUROC (OvR) | 0.9196 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7806 |
| Balanced Acc | 0.7835 |
| Macro-Sensitivity | 0.7835 |
| Macro-Specificity | 0.8887 |
| Macro-Precision | 0.7847 |
| AUC-PR (Macro) | 0.8527 |
| Cohen's Kappa | 0.6680 |
| MCC | 0.6685 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3836.3 (0.26 ms/张) |
| End-to-End FPS | 214.4 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.7344 | 0.8651 | 0.7705 | 0.7520 | 0.8407 | 0.8995 | 0.8475 | 256 |
| Malignant_lesions | 0.7429 | 0.8438 | 0.6842 | 0.7123 | 0.8781 | 0.8817 | 0.7563 | 210 |
| body_of_stomach_normal | 0.8732 | 0.9571 | 0.8995 | 0.8861 | 0.9449 | 0.9776 | 0.9543 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=56 TN=359 FN=68
- **Malignant_lesions**: TP=156 FP=72 TN=389 FN=54
- **body_of_stomach_normal**: TP=179 FP=20 TN=446 FN=26

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 54 | 14 |
| Malignant_lesions | 48 | 156 | 6 |
| body_of_stomach_normal | 8 | 18 | 179 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.7705    0.7344    0.7520       256
     Malignant_lesions     0.6842    0.7429    0.7123       210
body_of_stomach_normal     0.8995    0.8732    0.8861       205

              accuracy                         0.7794       671
             macro avg     0.7847    0.7835    0.7835       671
          weighted avg     0.7829    0.7794    0.7806       671

```