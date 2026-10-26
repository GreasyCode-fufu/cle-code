# shufflenet_v2_x1_0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/shufflenet_v2_x1_0_seed44/shufflenet_v2_x1_0/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9282 |
| Accuracy | 0.9288 |
| Macro-AUROC (OvR) | 0.9825 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9286 |
| Balanced Acc | 0.9266 |
| Macro-Sensitivity | 0.9266 |
| Macro-Specificity | 0.9638 |
| Macro-Precision | 0.9314 |
| AUC-PR (Macro) | 0.9695 |
| Cohen's Kappa | 0.8928 |
| MCC | 0.8935 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 151.69M |
| FLOPs | 303.38M |
| Params | 1.26M |
| Pure-Inference FPS | 6410.3 (0.16 ms/张) |
| End-to-End FPS | 257.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9321 | 0.8869 | 0.9138 | 0.9662 | 0.9770 | 0.9547 | 208 |
| Malignant_lesions | 0.8636 | 0.9750 | 0.9383 | 0.8994 | 0.9420 | 0.9736 | 0.9588 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9844 | 0.9689 | 0.9714 | 0.9869 | 0.9969 | 0.9950 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=25 TN=343 FN=12
- **Malignant_lesions**: TP=152 FP=10 TN=390 FN=24
- **body_of_stomach_normal**: TP=187 FP=6 TN=378 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 9 | 3 |
| Malignant_lesions | 21 | 152 | 3 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8869    0.9423    0.9138       208
     Malignant_lesions     0.9383    0.8636    0.8994       176
body_of_stomach_normal     0.9689    0.9740    0.9714       192

              accuracy                         0.9288       576
             macro avg     0.9314    0.9266    0.9282       576
          weighted avg     0.9299    0.9288    0.9286       576

```