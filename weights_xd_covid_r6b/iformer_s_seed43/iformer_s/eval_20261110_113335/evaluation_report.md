# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/iformer_s_seed43/iformer_s/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9777 |
| Accuracy | 0.9771 |
| Macro-AUROC (OvR) | 0.9976 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9771 |
| Balanced Acc | 0.9776 |
| Macro-Sensitivity | 0.9776 |
| Macro-Specificity | 0.9883 |
| Macro-Precision | 0.9780 |
| AUC-PR (Macro) | 0.9946 |
| Cohen's Kappa | 0.9656 |
| MCC | 0.9657 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 2954.7 (0.34 ms/张) |
| End-to-End FPS | 669.2 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 0.9998 | 0.9996 | 244 |
| NORMAL | 0.9594 | 0.9883 | 0.9774 | 0.9683 | 0.9788 | 0.9972 | 0.9888 | 271 |
| PNEUMONIA | 0.9815 | 0.9767 | 0.9567 | 0.9689 | 0.9902 | 0.9957 | 0.9953 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=260 FP=6 TN=508 FN=11
- **PNEUMONIA**: TP=265 FP=12 TN=503 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 260 | 11 |
| PNEUMONIA | 0 | 5 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9774    0.9594    0.9683       271
   PNEUMONIA     0.9567    0.9815    0.9689       270

    accuracy                         0.9771       785
   macro avg     0.9780    0.9776    0.9777       785
weighted avg     0.9773    0.9771    0.9771       785

```