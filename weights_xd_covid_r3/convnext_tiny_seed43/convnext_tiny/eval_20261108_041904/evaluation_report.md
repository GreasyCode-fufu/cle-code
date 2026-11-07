# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r3/convnext_tiny_seed43/convnext_tiny/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9825 |
| Accuracy | 0.9822 |
| Macro-AUROC (OvR) | 0.9991 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9822 |
| Balanced Acc | 0.9824 |
| Macro-Sensitivity | 0.9824 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9830 |
| AUC-PR (Macro) | 0.9983 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9734 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 901.5 (1.11 ms/张) |
| End-to-End FPS | 456.9 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9877 | 0.9982 | 0.9959 | 0.9918 | 0.9945 | 0.9998 | 0.9995 | 244 |
| NORMAL | 0.9668 | 0.9961 | 0.9924 | 0.9794 | 0.9827 | 0.9988 | 0.9974 | 271 |
| PNEUMONIA | 0.9926 | 0.9786 | 0.9606 | 0.9763 | 0.9960 | 0.9988 | 0.9980 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=241 FP=1 TN=540 FN=3
- **NORMAL**: TP=262 FP=2 TN=512 FN=9
- **PNEUMONIA**: TP=268 FP=11 TN=504 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 241 | 1 | 2 |
| NORMAL | 0 | 262 | 9 |
| PNEUMONIA | 1 | 1 | 268 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9877    0.9918       244
      NORMAL     0.9924    0.9668    0.9794       271
   PNEUMONIA     0.9606    0.9926    0.9763       270

    accuracy                         0.9822       785
   macro avg     0.9830    0.9824    0.9825       785
weighted avg     0.9825    0.9822    0.9822       785

```