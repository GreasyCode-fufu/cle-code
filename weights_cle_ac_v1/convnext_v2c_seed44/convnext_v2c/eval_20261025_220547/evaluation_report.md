# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2c_seed44/convnext_v2c/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9652 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9966 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9653 |
| Balanced Acc | 0.9651 |
| Macro-Sensitivity | 0.9651 |
| Macro-Specificity | 0.9825 |
| Macro-Precision | 0.9653 |
| AUC-PR (Macro) | 0.9939 |
| Cohen's Kappa | 0.9478 |
| MCC | 0.9478 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 847.0 (1.18 ms/张) |
| End-to-End FPS | 226.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9728 | 0.9522 | 0.9544 | 0.9755 | 0.9942 | 0.9919 | 208 |
| Malignant_lesions | 0.9489 | 0.9800 | 0.9543 | 0.9516 | 0.9776 | 0.9961 | 0.9910 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9948 | 0.9896 | 0.9896 | 0.9948 | 0.9994 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=10 TN=358 FN=9
- **Malignant_lesions**: TP=167 FP=8 TN=392 FN=9
- **body_of_stomach_normal**: TP=190 FP=2 TN=382 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 8 | 167 | 1 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9522    0.9567    0.9544       208
     Malignant_lesions     0.9543    0.9489    0.9516       176
body_of_stomach_normal     0.9896    0.9896    0.9896       192

              accuracy                         0.9653       576
             macro avg     0.9653    0.9651    0.9652       576
          weighted avg     0.9653    0.9653    0.9653       576

```