# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6/mbv3_dce4m_seed43/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9888 |
| Accuracy | 0.9885 |
| Macro-AUROC (OvR) | 0.9991 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9885 |
| Balanced Acc | 0.9888 |
| Macro-Sensitivity | 0.9888 |
| Macro-Specificity | 0.9942 |
| Macro-Precision | 0.9888 |
| AUC-PR (Macro) | 0.9984 |
| Cohen's Kappa | 0.9828 |
| MCC | 0.9828 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4172.3 (0.24 ms/张) |
| End-to-End FPS | 16.5 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9982 | 0.9959 | 0.9959 | 0.9982 | 0.9999 | 0.9997 | 244 |
| NORMAL | 0.9815 | 0.9942 | 0.9888 | 0.9852 | 0.9903 | 0.9990 | 0.9978 | 271 |
| PNEUMONIA | 0.9889 | 0.9903 | 0.9816 | 0.9852 | 0.9942 | 0.9983 | 0.9976 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=540 FN=1
- **NORMAL**: TP=266 FP=3 TN=511 FN=5
- **PNEUMONIA**: TP=267 FP=5 TN=510 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 1 | 266 | 4 |
| PNEUMONIA | 0 | 3 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9888    0.9815    0.9852       271
   PNEUMONIA     0.9816    0.9889    0.9852       270

    accuracy                         0.9885       785
   macro avg     0.9888    0.9888    0.9888       785
weighted avg     0.9886    0.9885    0.9885       785

```