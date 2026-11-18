# convnext_cmp 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_cmp_seed43/convnext_cmp/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9825 |
| Accuracy | 0.9822 |
| Macro-AUROC (OvR) | 0.9989 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9822 |
| Balanced Acc | 0.9823 |
| Macro-Sensitivity | 0.9823 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9827 |
| AUC-PR (Macro) | 0.9980 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 479.7 (2.08 ms/张) |
| End-to-End FPS | 314.8 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9877 | 0.9982 | 0.9959 | 0.9918 | 0.9945 | 0.9992 | 0.9988 | 244 |
| NORMAL | 0.9815 | 0.9903 | 0.9815 | 0.9815 | 0.9903 | 0.9991 | 0.9984 | 271 |
| PNEUMONIA | 0.9778 | 0.9845 | 0.9706 | 0.9742 | 0.9883 | 0.9984 | 0.9970 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=241 FP=1 TN=540 FN=3
- **NORMAL**: TP=266 FP=5 TN=509 FN=5
- **PNEUMONIA**: TP=264 FP=8 TN=507 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 241 | 0 | 3 |
| NORMAL | 0 | 266 | 5 |
| PNEUMONIA | 1 | 5 | 264 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9877    0.9918       244
      NORMAL     0.9815    0.9815    0.9815       271
   PNEUMONIA     0.9706    0.9778    0.9742       270

    accuracy                         0.9822       785
   macro avg     0.9827    0.9823    0.9825       785
weighted avg     0.9822    0.9822    0.9822       785

```