# convnext_v5xrcw 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5xrcw_seed42/convnext_v5xrcw/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9566 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9566 |
| Balanced Acc | 0.9552 |
| Macro-Sensitivity | 0.9552 |
| Macro-Specificity | 0.9778 |
| Macro-Precision | 0.9589 |
| AUC-PR (Macro) | 0.9907 |
| Cohen's Kappa | 0.9347 |
| MCC | 0.9351 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.69M |
| Pure-Inference FPS | 831.6 (1.20 ms/张) |
| End-to-End FPS | 212.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9538 | 0.9224 | 0.9461 | 0.9832 | 0.9936 | 0.9892 | 208 |
| Malignant_lesions | 0.9205 | 0.9875 | 0.9701 | 0.9446 | 0.9658 | 0.9936 | 0.9860 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9922 | 0.9842 | 0.9791 | 0.9870 | 0.9975 | 0.9969 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=17 TN=351 FN=6
- **Malignant_lesions**: TP=162 FP=5 TN=395 FN=14
- **body_of_stomach_normal**: TP=187 FP=3 TN=381 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 12 | 162 | 2 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9224    0.9712    0.9461       208
     Malignant_lesions     0.9701    0.9205    0.9446       176
body_of_stomach_normal     0.9842    0.9740    0.9791       192

              accuracy                         0.9566       576
             macro avg     0.9589    0.9552    0.9566       576
          weighted avg     0.9576    0.9566    0.9566       576

```