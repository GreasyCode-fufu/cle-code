# convnext_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_dsd_seed43/convnext_dsd/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9801 |
| Accuracy | 0.9796 |
| Macro-AUROC (OvR) | 0.9992 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9797 |
| Balanced Acc | 0.9798 |
| Macro-Sensitivity | 0.9798 |
| Macro-Specificity | 0.9896 |
| Macro-Precision | 0.9806 |
| AUC-PR (Macro) | 0.9985 |
| Cohen's Kappa | 0.9694 |
| MCC | 0.9695 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 456.8 (2.19 ms/张) |
| End-to-End FPS | 305.9 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9836 | 1.0000 | 1.0000 | 0.9917 | 0.9927 | 0.9999 | 0.9999 | 244 |
| NORMAL | 0.9668 | 0.9903 | 0.9813 | 0.9740 | 0.9826 | 0.9988 | 0.9976 | 271 |
| PNEUMONIA | 0.9889 | 0.9786 | 0.9604 | 0.9745 | 0.9941 | 0.9988 | 0.9979 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=240 FP=0 TN=541 FN=4
- **NORMAL**: TP=262 FP=5 TN=509 FN=9
- **PNEUMONIA**: TP=267 FP=11 TN=504 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 240 | 2 | 2 |
| NORMAL | 0 | 262 | 9 |
| PNEUMONIA | 0 | 3 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9836    0.9917       244
      NORMAL     0.9813    0.9668    0.9740       271
   PNEUMONIA     0.9604    0.9889    0.9745       270

    accuracy                         0.9796       785
   macro avg     0.9806    0.9798    0.9801       785
weighted avg     0.9799    0.9796    0.9797       785

```