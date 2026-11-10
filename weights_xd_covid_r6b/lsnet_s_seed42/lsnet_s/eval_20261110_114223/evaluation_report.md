# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/lsnet_s_seed42/lsnet_s/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9813 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9813 |
| Macro-Sensitivity | 0.9813 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9814 |
| AUC-PR (Macro) | 0.9945 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9713 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 4732.0 (0.21 ms/张) |
| End-to-End FPS | 745.0 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9963 | 0.9918 | 0.9939 | 0.9981 | 0.9981 | 0.9978 | 244 |
| NORMAL | 0.9667 | 0.9942 | 0.9886 | 0.9775 | 0.9827 | 0.9967 | 0.9958 | 270 |
| PNEUMONIA | 0.9815 | 0.9805 | 0.9636 | 0.9725 | 0.9902 | 0.9958 | 0.9899 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=2 TN=538 FN=1
- **NORMAL**: TP=261 FP=3 TN=511 FN=9
- **PNEUMONIA**: TP=265 FP=10 TN=504 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 261 | 9 |
| PNEUMONIA | 2 | 3 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9918    0.9959    0.9939       244
      NORMAL     0.9886    0.9667    0.9775       270
   PNEUMONIA     0.9636    0.9815    0.9725       270

    accuracy                         0.9809       784
   macro avg     0.9814    0.9813    0.9813       784
weighted avg     0.9810    0.9809    0.9809       784

```