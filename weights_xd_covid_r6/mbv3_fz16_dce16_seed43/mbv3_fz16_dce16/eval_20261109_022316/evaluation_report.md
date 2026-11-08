# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6/mbv3_fz16_dce16_seed43/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9826 |
| Accuracy | 0.9821 |
| Macro-AUROC (OvR) | 0.9970 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9821 |
| Balanced Acc | 0.9826 |
| Macro-Sensitivity | 0.9826 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9826 |
| AUC-PR (Macro) | 0.9941 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 4211.5 (0.24 ms/张) |
| End-to-End FPS | 19.5 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9967 | 0.9973 | 244 |
| NORMAL | 0.9815 | 0.9864 | 0.9743 | 0.9779 | 0.9902 | 0.9982 | 0.9971 | 270 |
| PNEUMONIA | 0.9704 | 0.9883 | 0.9776 | 0.9740 | 0.9845 | 0.9962 | 0.9881 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=265 FP=7 TN=507 FN=5
- **PNEUMONIA**: TP=262 FP=6 TN=508 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 265 | 5 |
| PNEUMONIA | 1 | 7 | 262 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9743    0.9815    0.9779       270
   PNEUMONIA     0.9776    0.9704    0.9740       270

    accuracy                         0.9821       784
   macro avg     0.9826    0.9826    0.9826       784
weighted avg     0.9822    0.9821    0.9821       784

```