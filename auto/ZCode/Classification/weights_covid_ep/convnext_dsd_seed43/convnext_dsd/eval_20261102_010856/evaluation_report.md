# convnext_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_dsd_seed43/convnext_dsd/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9863 |
| Accuracy | 0.9860 |
| Macro-AUROC (OvR) | 0.9972 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9860 |
| Balanced Acc | 0.9863 |
| Macro-Sensitivity | 0.9863 |
| Macro-Specificity | 0.9929 |
| Macro-Precision | 0.9863 |
| AUC-PR (Macro) | 0.9929 |
| Cohen's Kappa | 0.9789 |
| MCC | 0.9789 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 441.9 (2.26 ms/张) |
| End-to-End FPS | 298.6 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9959 | 0.9971 | 244 |
| NORMAL | 0.9815 | 0.9922 | 0.9851 | 0.9833 | 0.9903 | 0.9988 | 0.9980 | 270 |
| PNEUMONIA | 0.9815 | 0.9883 | 0.9779 | 0.9797 | 0.9903 | 0.9970 | 0.9837 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=265 FP=4 TN=510 FN=5
- **PNEUMONIA**: TP=265 FP=6 TN=508 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 265 | 5 |
| PNEUMONIA | 1 | 4 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9851    0.9815    0.9833       270
   PNEUMONIA     0.9779    0.9815    0.9797       270

    accuracy                         0.9860       784
   macro avg     0.9863    0.9863    0.9863       784
weighted avg     0.9860    0.9860    0.9860       784

```