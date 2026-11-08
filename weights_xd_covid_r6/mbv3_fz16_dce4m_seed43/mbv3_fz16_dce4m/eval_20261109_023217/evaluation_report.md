# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6/mbv3_fz16_dce4m_seed43/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9838 |
| Accuracy | 0.9834 |
| Macro-AUROC (OvR) | 0.9971 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9834 |
| Balanced Acc | 0.9838 |
| Macro-Sensitivity | 0.9838 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9838 |
| AUC-PR (Macro) | 0.9927 |
| Cohen's Kappa | 0.9751 |
| MCC | 0.9751 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3361.0 (0.30 ms/张) |
| End-to-End FPS | 15.3 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9966 | 0.9972 | 244 |
| NORMAL | 0.9852 | 0.9864 | 0.9744 | 0.9797 | 0.9922 | 0.9985 | 0.9974 | 270 |
| PNEUMONIA | 0.9704 | 0.9903 | 0.9813 | 0.9758 | 0.9845 | 0.9961 | 0.9835 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=266 FP=7 TN=507 FN=4
- **PNEUMONIA**: TP=262 FP=5 TN=509 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 266 | 4 |
| PNEUMONIA | 1 | 7 | 262 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9744    0.9852    0.9797       270
   PNEUMONIA     0.9813    0.9704    0.9758       270

    accuracy                         0.9834       784
   macro avg     0.9838    0.9838    0.9838       784
weighted avg     0.9834    0.9834    0.9834       784

```