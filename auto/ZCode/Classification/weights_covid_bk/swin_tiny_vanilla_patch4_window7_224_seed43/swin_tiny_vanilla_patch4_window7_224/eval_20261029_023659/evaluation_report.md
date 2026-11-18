# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/swin_tiny_vanilla_patch4_window7_224_seed43/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9812 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9971 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9808 |
| Balanced Acc | 0.9813 |
| Macro-Sensitivity | 0.9813 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9813 |
| AUC-PR (Macro) | 0.9950 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9714 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1558.3 (0.64 ms/张) |
| End-to-End FPS | 585.6 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9963 | 0.9918 | 0.9939 | 0.9981 | 0.9964 | 0.9973 | 244 |
| NORMAL | 0.9889 | 0.9825 | 0.9674 | 0.9780 | 0.9941 | 0.9982 | 0.9967 | 270 |
| PNEUMONIA | 0.9593 | 0.9922 | 0.9848 | 0.9719 | 0.9789 | 0.9967 | 0.9911 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=2 TN=538 FN=1
- **NORMAL**: TP=267 FP=9 TN=505 FN=3
- **PNEUMONIA**: TP=259 FP=4 TN=510 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 267 | 3 |
| PNEUMONIA | 2 | 9 | 259 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9918    0.9959    0.9939       244
      NORMAL     0.9674    0.9889    0.9780       270
   PNEUMONIA     0.9848    0.9593    0.9719       270

    accuracy                         0.9809       784
   macro avg     0.9813    0.9813    0.9812       784
weighted avg     0.9810    0.9809    0.9808       784

```