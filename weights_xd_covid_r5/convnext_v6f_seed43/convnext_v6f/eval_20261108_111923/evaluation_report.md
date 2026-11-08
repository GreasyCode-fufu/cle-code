# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r5/convnext_v6f_seed43/convnext_v6f/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9838 |
| Accuracy | 0.9834 |
| Macro-AUROC (OvR) | 0.9976 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9834 |
| Balanced Acc | 0.9837 |
| Macro-Sensitivity | 0.9837 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9840 |
| AUC-PR (Macro) | 0.9963 |
| Cohen's Kappa | 0.9751 |
| MCC | 0.9751 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 577.9 (1.73 ms/张) |
| End-to-End FPS | 356.1 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 0.9966 | 0.9973 | 244 |
| NORMAL | 0.9815 | 0.9883 | 0.9779 | 0.9797 | 0.9903 | 0.9988 | 0.9978 | 270 |
| PNEUMONIA | 0.9778 | 0.9864 | 0.9742 | 0.9760 | 0.9883 | 0.9974 | 0.9938 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=540 FN=2
- **NORMAL**: TP=265 FP=6 TN=508 FN=5
- **PNEUMONIA**: TP=264 FP=7 TN=507 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 0 | 2 |
| NORMAL | 0 | 265 | 5 |
| PNEUMONIA | 0 | 6 | 264 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9779    0.9815    0.9797       270
   PNEUMONIA     0.9742    0.9778    0.9760       270

    accuracy                         0.9834       784
   macro avg     0.9840    0.9837    0.9838       784
weighted avg     0.9835    0.9834    0.9834       784

```