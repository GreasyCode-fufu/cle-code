# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/mobilenet_v3_large_seed44/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9752 |
| Accuracy | 0.9745 |
| Macro-AUROC (OvR) | 0.9989 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9746 |
| Balanced Acc | 0.9750 |
| Macro-Sensitivity | 0.9750 |
| Macro-Specificity | 0.9871 |
| Macro-Precision | 0.9763 |
| AUC-PR (Macro) | 0.9980 |
| Cohen's Kappa | 0.9617 |
| MCC | 0.9622 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 5614.7 (0.18 ms/张) |
| End-to-End FPS | 798.3 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9877 | 1.0000 | 1.0000 | 0.9938 | 0.9945 | 0.9996 | 0.9993 | 244 |
| NORMAL | 0.9446 | 0.9942 | 0.9884 | 0.9660 | 0.9715 | 0.9987 | 0.9976 | 271 |
| PNEUMONIA | 0.9926 | 0.9670 | 0.9404 | 0.9658 | 0.9960 | 0.9985 | 0.9970 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=241 FP=0 TN=541 FN=3
- **NORMAL**: TP=256 FP=3 TN=511 FN=15
- **PNEUMONIA**: TP=268 FP=17 TN=498 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 241 | 1 | 2 |
| NORMAL | 0 | 256 | 15 |
| PNEUMONIA | 0 | 2 | 268 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9877    0.9938       244
      NORMAL     0.9884    0.9446    0.9660       271
   PNEUMONIA     0.9404    0.9926    0.9658       270

    accuracy                         0.9745       785
   macro avg     0.9763    0.9750    0.9752       785
weighted avg     0.9755    0.9745    0.9746       785

```