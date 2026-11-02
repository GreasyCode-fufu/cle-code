# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_svd_seed44/convnext_svd/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9851 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9991 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9849 |
| Macro-Sensitivity | 0.9849 |
| Macro-Specificity | 0.9922 |
| Macro-Precision | 0.9854 |
| AUC-PR (Macro) | 0.9984 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9771 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 1860.1 (0.54 ms/张) |
| End-to-End FPS | 616.0 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 0.9994 | 0.9990 | 244 |
| NORMAL | 0.9926 | 0.9844 | 0.9711 | 0.9818 | 0.9961 | 0.9994 | 0.9988 | 271 |
| PNEUMONIA | 0.9704 | 0.9922 | 0.9850 | 0.9776 | 0.9846 | 0.9987 | 0.9976 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=269 FP=8 TN=506 FN=2
- **PNEUMONIA**: TP=262 FP=4 TN=511 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 0 | 2 |
| NORMAL | 0 | 269 | 2 |
| PNEUMONIA | 0 | 8 | 262 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9711    0.9926    0.9818       271
   PNEUMONIA     0.9850    0.9704    0.9776       270

    accuracy                         0.9847       785
   macro avg     0.9854    0.9849    0.9851       785
weighted avg     0.9849    0.9847    0.9847       785

```