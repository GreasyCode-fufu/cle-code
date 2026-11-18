# convnext_dpea2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_dpea2_seed44/convnext_dpea2/model_best.pth`
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
| Macro-Precision | 0.9865 |
| AUC-PR (Macro) | 0.9991 |
| Cohen's Kappa | 0.9790 |
| MCC | 0.9790 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 28.12M |
| Pure-Inference FPS | 396.7 (2.52 ms/张) |
| End-to-End FPS | 274.4 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 1.0000 | 1.0000 | 244 |
| NORMAL | 0.9779 | 0.9922 | 0.9851 | 0.9815 | 0.9884 | 0.9993 | 0.9987 | 271 |
| PNEUMONIA | 0.9889 | 0.9864 | 0.9745 | 0.9816 | 0.9941 | 0.9993 | 0.9986 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=265 FP=4 TN=510 FN=6
- **PNEUMONIA**: TP=267 FP=7 TN=508 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 265 | 6 |
| PNEUMONIA | 0 | 3 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9851    0.9779    0.9815       271
   PNEUMONIA     0.9745    0.9889    0.9816       270

    accuracy                         0.9860       785
   macro avg     0.9865    0.9862    0.9863       785
weighted avg     0.9861    0.9860    0.9860       785

```