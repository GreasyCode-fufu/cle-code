# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_ef_seed42/convnext_ef/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9851 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9850 |
| Macro-Sensitivity | 0.9850 |
| Macro-Specificity | 0.9922 |
| Macro-Precision | 0.9853 |
| AUC-PR (Macro) | 0.9987 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9771 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 433.2 (2.31 ms/张) |
| End-to-End FPS | 293.7 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 1.0000 | 1.0000 | 244 |
| NORMAL | 0.9742 | 0.9922 | 0.9851 | 0.9796 | 0.9865 | 0.9990 | 0.9980 | 271 |
| PNEUMONIA | 0.9889 | 0.9845 | 0.9709 | 0.9798 | 0.9941 | 0.9989 | 0.9982 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=264 FP=4 TN=510 FN=7
- **PNEUMONIA**: TP=267 FP=8 TN=507 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 264 | 7 |
| PNEUMONIA | 0 | 3 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9851    0.9742    0.9796       271
   PNEUMONIA     0.9709    0.9889    0.9798       270

    accuracy                         0.9847       785
   macro avg     0.9853    0.9850    0.9851       785
weighted avg     0.9848    0.9847    0.9847       785

```