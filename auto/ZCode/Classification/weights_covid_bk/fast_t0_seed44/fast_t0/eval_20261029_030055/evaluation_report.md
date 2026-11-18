# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/fast_t0_seed44/fast_t0/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9826 |
| Accuracy | 0.9822 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9822 |
| Balanced Acc | 0.9825 |
| Macro-Sensitivity | 0.9825 |
| Macro-Specificity | 0.9909 |
| Macro-Precision | 0.9833 |
| AUC-PR (Macro) | 0.9987 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9735 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 9839.5 (0.10 ms/张) |
| End-to-End FPS | 841.9 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9963 | 0.9767 | 0.9574 | 0.9765 | 0.9980 | 0.9991 | 0.9984 | 271 |
| PNEUMONIA | 0.9593 | 0.9961 | 0.9923 | 0.9755 | 0.9790 | 0.9988 | 0.9977 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=270 FP=12 TN=502 FN=1
- **PNEUMONIA**: TP=259 FP=2 TN=513 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 270 | 1 |
| PNEUMONIA | 0 | 11 | 259 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9574    0.9963    0.9765       271
   PNEUMONIA     0.9923    0.9593    0.9755       270

    accuracy                         0.9822       785
   macro avg     0.9833    0.9825    0.9826       785
weighted avg     0.9827    0.9822    0.9822       785

```