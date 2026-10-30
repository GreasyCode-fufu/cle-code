# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/resnet50_seed44/resnet50/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9900 |
| Accuracy | 0.9898 |
| Macro-AUROC (OvR) | 0.9997 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9898 |
| Balanced Acc | 0.9899 |
| Macro-Sensitivity | 0.9899 |
| Macro-Specificity | 0.9948 |
| Macro-Precision | 0.9902 |
| AUC-PR (Macro) | 0.9995 |
| Cohen's Kappa | 0.9847 |
| MCC | 0.9847 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 2153.7 (0.46 ms/张) |
| End-to-End FPS | 652.3 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9889 | 0.9922 | 0.9853 | 0.9871 | 0.9942 | 0.9995 | 0.9991 | 271 |
| PNEUMONIA | 0.9889 | 0.9922 | 0.9852 | 0.9871 | 0.9942 | 0.9997 | 0.9995 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=268 FP=4 TN=510 FN=3
- **PNEUMONIA**: TP=267 FP=4 TN=511 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 268 | 3 |
| PNEUMONIA | 0 | 3 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9853    0.9889    0.9871       271
   PNEUMONIA     0.9852    0.9889    0.9871       270

    accuracy                         0.9898       785
   macro avg     0.9902    0.9899    0.9900       785
weighted avg     0.9898    0.9898    0.9898       785

```