# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_abl/convnext_v2m_seed42/convnext_v2m/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9814 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9994 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9814 |
| Macro-Sensitivity | 0.9814 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9816 |
| AUC-PR (Macro) | 0.9989 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9714 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 641.7 (1.56 ms/张) |
| End-to-End FPS | 376.1 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9982 | 0.9959 | 0.9959 | 0.9982 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9631 | 0.9942 | 0.9886 | 0.9757 | 0.9808 | 0.9992 | 0.9986 | 271 |
| PNEUMONIA | 0.9852 | 0.9786 | 0.9603 | 0.9726 | 0.9921 | 0.9990 | 0.9983 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=540 FN=1
- **NORMAL**: TP=261 FP=3 TN=511 FN=10
- **PNEUMONIA**: TP=266 FP=11 TN=504 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 261 | 10 |
| PNEUMONIA | 1 | 3 | 266 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9886    0.9631    0.9757       271
   PNEUMONIA     0.9603    0.9852    0.9726       270

    accuracy                         0.9809       785
   macro avg     0.9816    0.9814    0.9814       785
weighted avg     0.9811    0.9809    0.9809       785

```