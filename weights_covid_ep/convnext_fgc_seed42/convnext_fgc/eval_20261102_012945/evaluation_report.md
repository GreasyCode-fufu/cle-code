# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_fgc_seed42/convnext_fgc/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9824 |
| Accuracy | 0.9822 |
| Macro-AUROC (OvR) | 0.9991 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9822 |
| Balanced Acc | 0.9822 |
| Macro-Sensitivity | 0.9822 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9828 |
| AUC-PR (Macro) | 0.9982 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9733 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 854.1 (1.17 ms/张) |
| End-to-End FPS | 440.9 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9836 | 0.9982 | 0.9959 | 0.9897 | 0.9926 | 0.9999 | 0.9997 | 244 |
| NORMAL | 0.9742 | 0.9942 | 0.9888 | 0.9814 | 0.9865 | 0.9987 | 0.9974 | 271 |
| PNEUMONIA | 0.9889 | 0.9806 | 0.9639 | 0.9762 | 0.9941 | 0.9986 | 0.9976 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=240 FP=1 TN=540 FN=4
- **NORMAL**: TP=264 FP=3 TN=511 FN=7
- **PNEUMONIA**: TP=267 FP=10 TN=505 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 240 | 1 | 3 |
| NORMAL | 0 | 264 | 7 |
| PNEUMONIA | 1 | 2 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9836    0.9897       244
      NORMAL     0.9888    0.9742    0.9814       271
   PNEUMONIA     0.9639    0.9889    0.9762       270

    accuracy                         0.9822       785
   macro avg     0.9828    0.9822    0.9824       785
weighted avg     0.9824    0.9822    0.9822       785

```