# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_abl/convnext_v2m_seed44/convnext_v2m/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9815 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9995 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9814 |
| Macro-Sensitivity | 0.9814 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9817 |
| AUC-PR (Macro) | 0.9991 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9714 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1685.7 (0.59 ms/张) |
| End-to-End FPS | 603.4 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 1.0000 | 1.0000 | 0.9979 | 0.9982 | 1.0000 | 1.0000 | 244 |
| NORMAL | 0.9815 | 0.9805 | 0.9638 | 0.9726 | 0.9902 | 0.9993 | 0.9988 | 271 |
| PNEUMONIA | 0.9667 | 0.9903 | 0.9812 | 0.9739 | 0.9827 | 0.9993 | 0.9986 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=0 TN=541 FN=1
- **NORMAL**: TP=266 FP=10 TN=504 FN=5
- **PNEUMONIA**: TP=261 FP=5 TN=510 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 1 | 0 |
| NORMAL | 0 | 266 | 5 |
| PNEUMONIA | 0 | 9 | 261 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9959    0.9979       244
      NORMAL     0.9638    0.9815    0.9726       271
   PNEUMONIA     0.9812    0.9667    0.9739       270

    accuracy                         0.9809       785
   macro avg     0.9817    0.9814    0.9815       785
weighted avg     0.9810    0.9809    0.9809       785

```