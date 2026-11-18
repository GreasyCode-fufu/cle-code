# convnext_dpea2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_dpea2_seed43/convnext_dpea2/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9925 |
| Accuracy | 0.9924 |
| Macro-AUROC (OvR) | 0.9992 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9924 |
| Balanced Acc | 0.9925 |
| Macro-Sensitivity | 0.9925 |
| Macro-Specificity | 0.9961 |
| Macro-Precision | 0.9926 |
| AUC-PR (Macro) | 0.9985 |
| Cohen's Kappa | 0.9885 |
| MCC | 0.9885 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 28.12M |
| Pure-Inference FPS | 295.0 (3.39 ms/张) |
| End-to-End FPS | 223.2 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 1.0000 | 1.0000 | 0.9979 | 0.9982 | 1.0000 | 1.0000 | 244 |
| NORMAL | 0.9926 | 0.9942 | 0.9890 | 0.9908 | 0.9961 | 0.9989 | 0.9977 | 271 |
| PNEUMONIA | 0.9889 | 0.9942 | 0.9889 | 0.9889 | 0.9942 | 0.9985 | 0.9978 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=0 TN=541 FN=1
- **NORMAL**: TP=269 FP=3 TN=511 FN=2
- **PNEUMONIA**: TP=267 FP=3 TN=512 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 269 | 2 |
| PNEUMONIA | 0 | 3 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9959    0.9979       244
      NORMAL     0.9890    0.9926    0.9908       271
   PNEUMONIA     0.9889    0.9889    0.9889       270

    accuracy                         0.9924       785
   macro avg     0.9926    0.9925    0.9925       785
weighted avg     0.9924    0.9924    0.9924       785

```