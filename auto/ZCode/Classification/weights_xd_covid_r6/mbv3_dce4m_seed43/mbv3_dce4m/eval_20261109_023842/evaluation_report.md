# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6/mbv3_dce4m_seed43/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9813 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9977 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9813 |
| Macro-Sensitivity | 0.9813 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9814 |
| AUC-PR (Macro) | 0.9958 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9713 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3710.5 (0.27 ms/张) |
| End-to-End FPS | 16.6 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9973 | 0.9974 | 244 |
| NORMAL | 0.9815 | 0.9844 | 0.9707 | 0.9761 | 0.9902 | 0.9988 | 0.9978 | 270 |
| PNEUMONIA | 0.9667 | 0.9883 | 0.9775 | 0.9721 | 0.9826 | 0.9969 | 0.9923 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=265 FP=8 TN=506 FN=5
- **PNEUMONIA**: TP=261 FP=6 TN=508 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 265 | 5 |
| PNEUMONIA | 1 | 8 | 261 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9707    0.9815    0.9761       270
   PNEUMONIA     0.9775    0.9667    0.9721       270

    accuracy                         0.9809       784
   macro avg     0.9814    0.9813    0.9813       784
weighted avg     0.9809    0.9809    0.9809       784

```