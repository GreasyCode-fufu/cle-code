# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/resnet50_seed42/resnet50/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9814 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9991 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9814 |
| Macro-Sensitivity | 0.9814 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9821 |
| AUC-PR (Macro) | 0.9984 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9716 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 2155.7 (0.46 ms/张) |
| End-to-End FPS | 657.5 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 1.0000 | 1.0000 | 0.9979 | 0.9982 | 1.0000 | 1.0000 | 244 |
| NORMAL | 0.9520 | 0.9961 | 0.9923 | 0.9718 | 0.9752 | 0.9984 | 0.9972 | 271 |
| PNEUMONIA | 0.9963 | 0.9748 | 0.9539 | 0.9746 | 0.9980 | 0.9988 | 0.9979 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=0 TN=541 FN=1
- **NORMAL**: TP=258 FP=2 TN=512 FN=13
- **PNEUMONIA**: TP=269 FP=13 TN=502 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 1 | 0 |
| NORMAL | 0 | 258 | 13 |
| PNEUMONIA | 0 | 1 | 269 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9959    0.9979       244
      NORMAL     0.9923    0.9520    0.9718       271
   PNEUMONIA     0.9539    0.9963    0.9746       270

    accuracy                         0.9809       785
   macro avg     0.9821    0.9814    0.9814       785
weighted avg     0.9815    0.9809    0.9809       785

```