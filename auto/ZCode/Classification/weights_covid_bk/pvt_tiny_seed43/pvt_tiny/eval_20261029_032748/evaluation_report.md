# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/pvt_tiny_seed43/pvt_tiny/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9836 |
| Accuracy | 0.9834 |
| Macro-AUROC (OvR) | 0.9982 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9834 |
| Balanced Acc | 0.9837 |
| Macro-Sensitivity | 0.9837 |
| Macro-Specificity | 0.9917 |
| Macro-Precision | 0.9836 |
| AUC-PR (Macro) | 0.9965 |
| Cohen's Kappa | 0.9751 |
| MCC | 0.9751 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2941.4 (0.34 ms/张) |
| End-to-End FPS | 709.2 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9944 | 0.9878 | 0.9898 | 0.9963 | 0.9987 | 0.9983 | 244 |
| NORMAL | 0.9889 | 0.9883 | 0.9780 | 0.9834 | 0.9941 | 0.9987 | 0.9977 | 270 |
| PNEUMONIA | 0.9704 | 0.9922 | 0.9850 | 0.9776 | 0.9846 | 0.9972 | 0.9935 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=3 TN=537 FN=2
- **NORMAL**: TP=267 FP=6 TN=508 FN=3
- **PNEUMONIA**: TP=262 FP=4 TN=510 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 267 | 3 |
| PNEUMONIA | 3 | 5 | 262 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9878    0.9918    0.9898       244
      NORMAL     0.9780    0.9889    0.9834       270
   PNEUMONIA     0.9850    0.9704    0.9776       270

    accuracy                         0.9834       784
   macro avg     0.9836    0.9837    0.9836       784
weighted avg     0.9834    0.9834    0.9834       784

```