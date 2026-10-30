# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/repvit_m1_seed43/repvit_m1/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9625 |
| Accuracy | 0.9617 |
| Macro-AUROC (OvR) | 0.9941 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9617 |
| Balanced Acc | 0.9626 |
| Macro-Sensitivity | 0.9626 |
| Macro-Specificity | 0.9807 |
| Macro-Precision | 0.9624 |
| AUC-PR (Macro) | 0.9880 |
| Cohen's Kappa | 0.9425 |
| MCC | 0.9426 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3881.8 (0.26 ms/张) |
| End-to-End FPS | 747.8 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9877 | 0.9926 | 0.9837 | 0.9857 | 0.9944 | 0.9967 | 0.9958 | 244 |
| NORMAL | 0.9481 | 0.9786 | 0.9588 | 0.9534 | 0.9729 | 0.9948 | 0.9900 | 270 |
| PNEUMONIA | 0.9519 | 0.9708 | 0.9449 | 0.9483 | 0.9746 | 0.9907 | 0.9780 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=241 FP=4 TN=536 FN=3
- **NORMAL**: TP=256 FP=11 TN=503 FN=14
- **PNEUMONIA**: TP=257 FP=15 TN=499 FN=13

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 241 | 2 | 1 |
| NORMAL | 0 | 256 | 14 |
| PNEUMONIA | 4 | 9 | 257 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9837    0.9877    0.9857       244
      NORMAL     0.9588    0.9481    0.9534       270
   PNEUMONIA     0.9449    0.9519    0.9483       270

    accuracy                         0.9617       784
   macro avg     0.9624    0.9626    0.9625       784
weighted avg     0.9617    0.9617    0.9617       784

```