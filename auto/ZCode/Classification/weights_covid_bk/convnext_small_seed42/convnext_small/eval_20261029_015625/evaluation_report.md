# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/convnext_small_seed42/convnext_small/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9825 |
| Accuracy | 0.9822 |
| Macro-AUROC (OvR) | 0.9996 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9822 |
| Balanced Acc | 0.9825 |
| Macro-Sensitivity | 0.9825 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9825 |
| AUC-PR (Macro) | 0.9992 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 1059.7 (0.94 ms/张) |
| End-to-End FPS | 499.6 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9963 | 0.9918 | 0.9918 | 0.9963 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9815 | 0.9883 | 0.9779 | 0.9797 | 0.9903 | 0.9994 | 0.9990 | 271 |
| PNEUMONIA | 0.9741 | 0.9883 | 0.9777 | 0.9759 | 0.9864 | 0.9994 | 0.9989 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=2 TN=539 FN=2
- **NORMAL**: TP=266 FP=6 TN=508 FN=5
- **PNEUMONIA**: TP=263 FP=6 TN=509 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 266 | 5 |
| PNEUMONIA | 2 | 5 | 263 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9918    0.9918    0.9918       244
      NORMAL     0.9779    0.9815    0.9797       271
   PNEUMONIA     0.9777    0.9741    0.9759       270

    accuracy                         0.9822       785
   macro avg     0.9825    0.9825    0.9825       785
weighted avg     0.9822    0.9822    0.9822       785

```