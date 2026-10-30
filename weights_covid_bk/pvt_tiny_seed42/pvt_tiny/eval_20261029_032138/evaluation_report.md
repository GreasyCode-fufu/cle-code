# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/pvt_tiny_seed42/pvt_tiny/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9813 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9971 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9808 |
| Balanced Acc | 0.9813 |
| Macro-Sensitivity | 0.9813 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9812 |
| AUC-PR (Macro) | 0.9941 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9713 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2912.7 (0.34 ms/张) |
| End-to-End FPS | 717.0 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9963 | 0.9918 | 0.9939 | 0.9981 | 0.9990 | 0.9984 | 244 |
| NORMAL | 0.9815 | 0.9864 | 0.9743 | 0.9779 | 0.9902 | 0.9973 | 0.9950 | 270 |
| PNEUMONIA | 0.9667 | 0.9883 | 0.9775 | 0.9721 | 0.9826 | 0.9951 | 0.9890 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=2 TN=538 FN=1
- **NORMAL**: TP=265 FP=7 TN=507 FN=5
- **PNEUMONIA**: TP=261 FP=6 TN=508 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 265 | 5 |
| PNEUMONIA | 2 | 7 | 261 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9918    0.9959    0.9939       244
      NORMAL     0.9743    0.9815    0.9779       270
   PNEUMONIA     0.9775    0.9667    0.9721       270

    accuracy                         0.9809       784
   macro avg     0.9812    0.9813    0.9813       784
weighted avg     0.9809    0.9809    0.9808       784

```