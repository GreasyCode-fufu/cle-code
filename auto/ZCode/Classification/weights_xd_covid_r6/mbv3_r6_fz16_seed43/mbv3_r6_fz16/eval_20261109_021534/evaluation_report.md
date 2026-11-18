# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6/mbv3_r6_fz16_seed43/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9839 |
| Accuracy | 0.9834 |
| Macro-AUROC (OvR) | 0.9991 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9835 |
| Balanced Acc | 0.9837 |
| Macro-Sensitivity | 0.9837 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9840 |
| AUC-PR (Macro) | 0.9983 |
| Cohen's Kappa | 0.9751 |
| MCC | 0.9751 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3229.8 (0.31 ms/张) |
| End-to-End FPS | 507.2 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9779 | 0.9883 | 0.9779 | 0.9779 | 0.9883 | 0.9987 | 0.9975 | 271 |
| PNEUMONIA | 0.9815 | 0.9864 | 0.9743 | 0.9779 | 0.9903 | 0.9985 | 0.9974 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=265 FP=6 TN=508 FN=6
- **PNEUMONIA**: TP=265 FP=7 TN=508 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 265 | 6 |
| PNEUMONIA | 0 | 5 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9779    0.9779    0.9779       271
   PNEUMONIA     0.9743    0.9815    0.9779       270

    accuracy                         0.9834       785
   macro avg     0.9840    0.9837    0.9839       785
weighted avg     0.9835    0.9834    0.9835       785

```