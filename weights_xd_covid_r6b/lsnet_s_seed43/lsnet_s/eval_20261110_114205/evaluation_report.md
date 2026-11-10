# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/lsnet_s_seed43/lsnet_s/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9864 |
| Accuracy | 0.9860 |
| Macro-AUROC (OvR) | 0.9992 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9860 |
| Balanced Acc | 0.9863 |
| Macro-Sensitivity | 0.9863 |
| Macro-Specificity | 0.9929 |
| Macro-Precision | 0.9865 |
| AUC-PR (Macro) | 0.9986 |
| Cohen's Kappa | 0.9790 |
| MCC | 0.9790 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 3248.6 (0.31 ms/张) |
| End-to-End FPS | 694.6 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 1.0000 | 1.0000 | 0.9979 | 0.9982 | 0.9999 | 0.9999 | 244 |
| NORMAL | 0.9815 | 0.9903 | 0.9815 | 0.9815 | 0.9903 | 0.9990 | 0.9985 | 271 |
| PNEUMONIA | 0.9815 | 0.9883 | 0.9779 | 0.9797 | 0.9903 | 0.9988 | 0.9974 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=0 TN=541 FN=1
- **NORMAL**: TP=266 FP=5 TN=509 FN=5
- **PNEUMONIA**: TP=265 FP=6 TN=509 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 266 | 5 |
| PNEUMONIA | 0 | 5 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9959    0.9979       244
      NORMAL     0.9815    0.9815    0.9815       271
   PNEUMONIA     0.9779    0.9815    0.9797       270

    accuracy                         0.9860       785
   macro avg     0.9865    0.9863    0.9864       785
weighted avg     0.9860    0.9860    0.9860       785

```