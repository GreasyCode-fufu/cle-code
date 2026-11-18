# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/swin_tiny_vanilla_patch4_window7_224_seed43/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9800 |
| Accuracy | 0.9796 |
| Macro-AUROC (OvR) | 0.9992 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9796 |
| Balanced Acc | 0.9800 |
| Macro-Sensitivity | 0.9800 |
| Macro-Specificity | 0.9897 |
| Macro-Precision | 0.9802 |
| AUC-PR (Macro) | 0.9985 |
| Cohen's Kappa | 0.9694 |
| MCC | 0.9695 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1553.6 (0.64 ms/张) |
| End-to-End FPS | 584.8 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9963 | 0.9918 | 0.9918 | 0.9963 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9889 | 0.9805 | 0.9640 | 0.9763 | 0.9941 | 0.9989 | 0.9979 | 271 |
| PNEUMONIA | 0.9593 | 0.9922 | 0.9848 | 0.9719 | 0.9789 | 0.9988 | 0.9978 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=2 TN=539 FN=2
- **NORMAL**: TP=268 FP=10 TN=504 FN=3
- **PNEUMONIA**: TP=259 FP=4 TN=511 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 268 | 3 |
| PNEUMONIA | 2 | 9 | 259 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9918    0.9918    0.9918       244
      NORMAL     0.9640    0.9889    0.9763       271
   PNEUMONIA     0.9848    0.9593    0.9719       270

    accuracy                         0.9796       785
   macro avg     0.9802    0.9800    0.9800       785
weighted avg     0.9798    0.9796    0.9796       785

```