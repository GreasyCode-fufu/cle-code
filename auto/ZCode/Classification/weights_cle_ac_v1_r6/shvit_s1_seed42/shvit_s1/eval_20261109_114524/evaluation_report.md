# shvit_s1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/shvit_s1_seed42/shvit_s1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9340 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9901 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9340 |
| Balanced Acc | 0.9323 |
| Macro-Sensitivity | 0.9323 |
| Macro-Specificity | 0.9664 |
| Macro-Precision | 0.9372 |
| AUC-PR (Macro) | 0.9810 |
| Cohen's Kappa | 0.9006 |
| MCC | 0.9013 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 245.71M |
| FLOPs | 491.41M |
| Params | 6.01M |
| Pure-Inference FPS | 4716.0 (0.21 ms/张) |
| End-to-End FPS | 205.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9348 | 0.8909 | 0.9159 | 0.9663 | 0.9847 | 0.9776 | 208 |
| Malignant_lesions | 0.8807 | 0.9825 | 0.9568 | 0.9172 | 0.9493 | 0.9874 | 0.9688 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9818 | 0.9639 | 0.9689 | 0.9869 | 0.9983 | 0.9966 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=24 TN=344 FN=12
- **Malignant_lesions**: TP=155 FP=7 TN=393 FN=21
- **body_of_stomach_normal**: TP=187 FP=7 TN=377 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 7 | 5 |
| Malignant_lesions | 19 | 155 | 2 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8909    0.9423    0.9159       208
     Malignant_lesions     0.9568    0.8807    0.9172       176
body_of_stomach_normal     0.9639    0.9740    0.9689       192

              accuracy                         0.9340       576
             macro avg     0.9372    0.9323    0.9340       576
          weighted avg     0.9354    0.9340    0.9340       576

```