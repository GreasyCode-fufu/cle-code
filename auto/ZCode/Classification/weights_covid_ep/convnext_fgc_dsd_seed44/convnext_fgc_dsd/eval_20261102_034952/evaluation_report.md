# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_fgc_dsd_seed44/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9802 |
| Accuracy | 0.9796 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9796 |
| Balanced Acc | 0.9802 |
| Macro-Sensitivity | 0.9802 |
| Macro-Specificity | 0.9896 |
| Macro-Precision | 0.9807 |
| AUC-PR (Macro) | 0.9986 |
| Cohen's Kappa | 0.9694 |
| MCC | 0.9696 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 495.5 (2.02 ms/张) |
| End-to-End FPS | 319.1 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 1.0000 | 1.0000 | 0.9979 | 0.9982 | 1.0000 | 1.0000 | 244 |
| NORMAL | 0.9557 | 0.9942 | 0.9885 | 0.9719 | 0.9771 | 0.9990 | 0.9981 | 271 |
| PNEUMONIA | 0.9889 | 0.9748 | 0.9536 | 0.9709 | 0.9941 | 0.9988 | 0.9978 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=0 TN=541 FN=1
- **NORMAL**: TP=259 FP=3 TN=511 FN=12
- **PNEUMONIA**: TP=267 FP=13 TN=502 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 259 | 12 |
| PNEUMONIA | 0 | 3 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9959    0.9979       244
      NORMAL     0.9885    0.9557    0.9719       271
   PNEUMONIA     0.9536    0.9889    0.9709       270

    accuracy                         0.9796       785
   macro avg     0.9807    0.9802    0.9802       785
weighted avg     0.9801    0.9796    0.9796       785

```