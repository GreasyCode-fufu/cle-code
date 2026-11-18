# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/resnet34_seed42/resnet34/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9837 |
| Accuracy | 0.9834 |
| Macro-AUROC (OvR) | 0.9974 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9834 |
| Balanced Acc | 0.9837 |
| Macro-Sensitivity | 0.9837 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9840 |
| AUC-PR (Macro) | 0.9955 |
| Cohen's Kappa | 0.9751 |
| MCC | 0.9752 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3811.1 (0.26 ms/张) |
| End-to-End FPS | 744.9 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9963 | 0.9918 | 0.9918 | 0.9963 | 0.9962 | 0.9971 | 244 |
| NORMAL | 0.9704 | 0.9981 | 0.9962 | 0.9831 | 0.9846 | 0.9991 | 0.9983 | 270 |
| PNEUMONIA | 0.9889 | 0.9805 | 0.9639 | 0.9762 | 0.9941 | 0.9968 | 0.9912 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=2 TN=538 FN=2
- **NORMAL**: TP=262 FP=1 TN=513 FN=8
- **PNEUMONIA**: TP=267 FP=10 TN=504 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 0 | 2 |
| NORMAL | 0 | 262 | 8 |
| PNEUMONIA | 2 | 1 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9918    0.9918    0.9918       244
      NORMAL     0.9962    0.9704    0.9831       270
   PNEUMONIA     0.9639    0.9889    0.9762       270

    accuracy                         0.9834       784
   macro avg     0.9840    0.9837    0.9837       784
weighted avg     0.9837    0.9834    0.9834       784

```