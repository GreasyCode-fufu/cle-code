# convnext_dph 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_dph_seed43/convnext_dph/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9850 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9972 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9851 |
| Macro-Sensitivity | 0.9851 |
| Macro-Specificity | 0.9922 |
| Macro-Precision | 0.9851 |
| AUC-PR (Macro) | 0.9952 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9771 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 511.8 (1.95 ms/张) |
| End-to-End FPS | 327.7 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9972 | 0.9973 | 244 |
| NORMAL | 0.9889 | 0.9864 | 0.9745 | 0.9816 | 0.9941 | 0.9981 | 0.9968 | 270 |
| PNEUMONIA | 0.9704 | 0.9922 | 0.9850 | 0.9776 | 0.9846 | 0.9964 | 0.9915 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=267 FP=7 TN=507 FN=3
- **PNEUMONIA**: TP=262 FP=4 TN=510 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 267 | 3 |
| PNEUMONIA | 1 | 7 | 262 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9745    0.9889    0.9816       270
   PNEUMONIA     0.9850    0.9704    0.9776       270

    accuracy                         0.9847       784
   macro avg     0.9851    0.9851    0.9850       784
weighted avg     0.9847    0.9847    0.9847       784

```