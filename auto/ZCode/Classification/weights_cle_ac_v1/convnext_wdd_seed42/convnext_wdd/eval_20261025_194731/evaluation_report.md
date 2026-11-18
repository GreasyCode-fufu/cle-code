# convnext_wdd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wdd_seed42/convnext_wdd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9672 |
| Accuracy | 0.9670 |
| Macro-AUROC (OvR) | 0.9973 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9671 |
| Balanced Acc | 0.9674 |
| Macro-Sensitivity | 0.9674 |
| Macro-Specificity | 0.9835 |
| Macro-Precision | 0.9671 |
| AUC-PR (Macro) | 0.9947 |
| Cohen's Kappa | 0.9504 |
| MCC | 0.9505 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 29.35M |
| Pure-Inference FPS | 654.4 (1.53 ms/张) |
| End-to-End FPS | 205.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9755 | 0.9567 | 0.9567 | 0.9755 | 0.9956 | 0.9936 | 208 |
| Malignant_lesions | 0.9716 | 0.9775 | 0.9500 | 0.9607 | 0.9874 | 0.9966 | 0.9913 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9974 | 0.9947 | 0.9842 | 0.9871 | 0.9995 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=9 TN=359 FN=9
- **Malignant_lesions**: TP=171 FP=9 TN=391 FN=5
- **body_of_stomach_normal**: TP=187 FP=1 TN=383 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 5 | 171 | 0 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9567    0.9567    0.9567       208
     Malignant_lesions     0.9500    0.9716    0.9607       176
body_of_stomach_normal     0.9947    0.9740    0.9842       192

              accuracy                         0.9670       576
             macro avg     0.9671    0.9674    0.9672       576
          weighted avg     0.9673    0.9670    0.9671       576

```