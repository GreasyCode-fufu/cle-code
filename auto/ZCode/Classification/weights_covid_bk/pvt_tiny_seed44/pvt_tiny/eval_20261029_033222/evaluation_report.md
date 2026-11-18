# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/pvt_tiny_seed44/pvt_tiny/model_best.pth`
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
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2923.5 (0.34 ms/张) |
| End-to-End FPS | 711.3 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 1.0000 | 1.0000 | 244 |
| NORMAL | 0.9815 | 0.9864 | 0.9744 | 0.9779 | 0.9902 | 0.9986 | 0.9973 | 271 |
| PNEUMONIA | 0.9778 | 0.9883 | 0.9778 | 0.9778 | 0.9883 | 0.9986 | 0.9976 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=266 FP=7 TN=507 FN=5
- **PNEUMONIA**: TP=264 FP=6 TN=509 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 266 | 5 |
| PNEUMONIA | 0 | 6 | 264 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9744    0.9815    0.9779       271
   PNEUMONIA     0.9778    0.9778    0.9778       270

    accuracy                         0.9834       785
   macro avg     0.9840    0.9837    0.9839       785
weighted avg     0.9835    0.9834    0.9835       785

```