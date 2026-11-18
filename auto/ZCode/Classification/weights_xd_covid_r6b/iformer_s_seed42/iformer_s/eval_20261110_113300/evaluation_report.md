# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/iformer_s_seed42/iformer_s/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9851 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9995 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9850 |
| Macro-Sensitivity | 0.9850 |
| Macro-Specificity | 0.9922 |
| Macro-Precision | 0.9854 |
| AUC-PR (Macro) | 0.9990 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9771 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 3085.9 (0.32 ms/张) |
| End-to-End FPS | 694.7 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9705 | 0.9942 | 0.9887 | 0.9795 | 0.9846 | 0.9992 | 0.9985 | 271 |
| PNEUMONIA | 0.9926 | 0.9825 | 0.9675 | 0.9799 | 0.9961 | 0.9993 | 0.9987 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=263 FP=3 TN=511 FN=8
- **PNEUMONIA**: TP=268 FP=9 TN=506 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 263 | 8 |
| PNEUMONIA | 0 | 2 | 268 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9887    0.9705    0.9795       271
   PNEUMONIA     0.9675    0.9926    0.9799       270

    accuracy                         0.9847       785
   macro avg     0.9854    0.9850    0.9851       785
weighted avg     0.9849    0.9847    0.9847       785

```