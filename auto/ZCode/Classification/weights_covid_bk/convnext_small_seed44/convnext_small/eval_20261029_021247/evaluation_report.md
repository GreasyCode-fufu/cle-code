# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/convnext_small_seed44/convnext_small/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9851 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9973 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9851 |
| Macro-Sensitivity | 0.9851 |
| Macro-Specificity | 0.9922 |
| Macro-Precision | 0.9852 |
| AUC-PR (Macro) | 0.9941 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9771 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 1058.4 (0.94 ms/张) |
| End-to-End FPS | 500.0 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9960 | 0.9971 | 244 |
| NORMAL | 0.9704 | 0.9961 | 0.9924 | 0.9813 | 0.9846 | 0.9990 | 0.9983 | 270 |
| PNEUMONIA | 0.9889 | 0.9825 | 0.9674 | 0.9780 | 0.9941 | 0.9968 | 0.9869 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=262 FP=2 TN=512 FN=8
- **PNEUMONIA**: TP=267 FP=9 TN=505 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 262 | 8 |
| PNEUMONIA | 1 | 2 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9924    0.9704    0.9813       270
   PNEUMONIA     0.9674    0.9889    0.9780       270

    accuracy                         0.9847       784
   macro avg     0.9852    0.9851    0.9851       784
weighted avg     0.9849    0.9847    0.9847       784

```