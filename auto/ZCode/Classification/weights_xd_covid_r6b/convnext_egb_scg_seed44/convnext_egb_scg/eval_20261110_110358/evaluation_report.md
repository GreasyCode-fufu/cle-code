# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/convnext_egb_scg_seed44/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9863 |
| Accuracy | 0.9860 |
| Macro-AUROC (OvR) | 0.9974 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9860 |
| Balanced Acc | 0.9863 |
| Macro-Sensitivity | 0.9863 |
| Macro-Specificity | 0.9929 |
| Macro-Precision | 0.9863 |
| AUC-PR (Macro) | 0.9955 |
| Cohen's Kappa | 0.9789 |
| MCC | 0.9790 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 368.9 (2.71 ms/张) |
| End-to-End FPS | 263.9 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9967 | 0.9973 | 244 |
| NORMAL | 0.9889 | 0.9883 | 0.9780 | 0.9834 | 0.9941 | 0.9985 | 0.9974 | 270 |
| PNEUMONIA | 0.9741 | 0.9922 | 0.9850 | 0.9795 | 0.9865 | 0.9970 | 0.9918 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=267 FP=6 TN=508 FN=3
- **PNEUMONIA**: TP=263 FP=4 TN=510 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 267 | 3 |
| PNEUMONIA | 1 | 6 | 263 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9780    0.9889    0.9834       270
   PNEUMONIA     0.9850    0.9741    0.9795       270

    accuracy                         0.9860       784
   macro avg     0.9863    0.9863    0.9863       784
weighted avg     0.9860    0.9860    0.9860       784

```