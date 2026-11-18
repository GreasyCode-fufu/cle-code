# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_fgc_seed44/convnext_fgc/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9863 |
| Accuracy | 0.9860 |
| Macro-AUROC (OvR) | 0.9995 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9860 |
| Balanced Acc | 0.9862 |
| Macro-Sensitivity | 0.9862 |
| Macro-Specificity | 0.9929 |
| Macro-Precision | 0.9864 |
| AUC-PR (Macro) | 0.9991 |
| Cohen's Kappa | 0.9790 |
| MCC | 0.9790 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 815.1 (1.23 ms/张) |
| End-to-End FPS | 429.1 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9982 | 0.9959 | 0.9938 | 0.9963 | 0.9999 | 0.9999 | 244 |
| NORMAL | 0.9815 | 0.9942 | 0.9888 | 0.9852 | 0.9903 | 0.9994 | 0.9989 | 271 |
| PNEUMONIA | 0.9852 | 0.9864 | 0.9744 | 0.9797 | 0.9922 | 0.9992 | 0.9986 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=1 TN=540 FN=2
- **NORMAL**: TP=266 FP=3 TN=511 FN=5
- **PNEUMONIA**: TP=266 FP=7 TN=508 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 0 | 2 |
| NORMAL | 0 | 266 | 5 |
| PNEUMONIA | 1 | 3 | 266 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9918    0.9938       244
      NORMAL     0.9888    0.9815    0.9852       271
   PNEUMONIA     0.9744    0.9852    0.9797       270

    accuracy                         0.9860       785
   macro avg     0.9864    0.9862    0.9863       785
weighted avg     0.9861    0.9860    0.9860       785

```