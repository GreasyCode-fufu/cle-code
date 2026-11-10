# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/jumbo_small_seed44/jumbo_small/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9826 |
| Accuracy | 0.9822 |
| Macro-AUROC (OvR) | 0.9988 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9822 |
| Balanced Acc | 0.9823 |
| Macro-Sensitivity | 0.9823 |
| Macro-Specificity | 0.9909 |
| Macro-Precision | 0.9830 |
| AUC-PR (Macro) | 0.9978 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9733 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 853.3 (1.17 ms/张) |
| End-to-End FPS | 443.8 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9877 | 1.0000 | 1.0000 | 0.9938 | 0.9945 | 0.9998 | 0.9996 | 244 |
| NORMAL | 0.9889 | 0.9805 | 0.9640 | 0.9763 | 0.9941 | 0.9980 | 0.9966 | 271 |
| PNEUMONIA | 0.9704 | 0.9922 | 0.9850 | 0.9776 | 0.9846 | 0.9985 | 0.9973 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=241 FP=0 TN=541 FN=3
- **NORMAL**: TP=268 FP=10 TN=504 FN=3
- **PNEUMONIA**: TP=262 FP=4 TN=511 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 241 | 2 | 1 |
| NORMAL | 0 | 268 | 3 |
| PNEUMONIA | 0 | 8 | 262 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9877    0.9938       244
      NORMAL     0.9640    0.9889    0.9763       271
   PNEUMONIA     0.9850    0.9704    0.9776       270

    accuracy                         0.9822       785
   macro avg     0.9830    0.9823    0.9826       785
weighted avg     0.9824    0.9822    0.9822       785

```