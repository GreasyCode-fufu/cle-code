# convnext_dph 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_dph_seed43/convnext_dph/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9851 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9992 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9849 |
| Macro-Sensitivity | 0.9849 |
| Macro-Specificity | 0.9922 |
| Macro-Precision | 0.9853 |
| AUC-PR (Macro) | 0.9986 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9771 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 581.0 (1.72 ms/张) |
| End-to-End FPS | 352.4 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9889 | 0.9844 | 0.9710 | 0.9799 | 0.9941 | 0.9990 | 0.9982 | 271 |
| PNEUMONIA | 0.9741 | 0.9922 | 0.9850 | 0.9795 | 0.9865 | 0.9987 | 0.9977 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=268 FP=8 TN=506 FN=3
- **PNEUMONIA**: TP=263 FP=4 TN=511 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 268 | 3 |
| PNEUMONIA | 0 | 7 | 263 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9710    0.9889    0.9799       271
   PNEUMONIA     0.9850    0.9741    0.9795       270

    accuracy                         0.9847       785
   macro avg     0.9853    0.9849    0.9851       785
weighted avg     0.9848    0.9847    0.9847       785

```