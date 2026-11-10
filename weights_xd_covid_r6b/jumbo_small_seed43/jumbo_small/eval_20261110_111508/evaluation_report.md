# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/jumbo_small_seed43/jumbo_small/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9815 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9985 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9814 |
| Macro-Sensitivity | 0.9814 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9816 |
| AUC-PR (Macro) | 0.9972 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9713 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 800.7 (1.25 ms/张) |
| End-to-End FPS | 426.1 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 1.0000 | 1.0000 | 0.9979 | 0.9982 | 0.9995 | 0.9991 | 244 |
| NORMAL | 0.9779 | 0.9844 | 0.9707 | 0.9743 | 0.9883 | 0.9981 | 0.9962 | 271 |
| PNEUMONIA | 0.9704 | 0.9864 | 0.9740 | 0.9722 | 0.9845 | 0.9978 | 0.9962 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=0 TN=541 FN=1
- **NORMAL**: TP=265 FP=8 TN=506 FN=6
- **PNEUMONIA**: TP=262 FP=7 TN=508 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 265 | 6 |
| PNEUMONIA | 0 | 8 | 262 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9959    0.9979       244
      NORMAL     0.9707    0.9779    0.9743       271
   PNEUMONIA     0.9740    0.9704    0.9722       270

    accuracy                         0.9809       785
   macro avg     0.9816    0.9814    0.9815       785
weighted avg     0.9809    0.9809    0.9809       785

```