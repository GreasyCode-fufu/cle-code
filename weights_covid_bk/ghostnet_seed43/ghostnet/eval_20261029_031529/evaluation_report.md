# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/ghostnet_seed43/ghostnet/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9813 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9975 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9813 |
| Macro-Sensitivity | 0.9813 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9814 |
| AUC-PR (Macro) | 0.9951 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9713 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 5099.9 (0.20 ms/张) |
| End-to-End FPS | 793.6 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9972 | 0.9974 | 244 |
| NORMAL | 0.9852 | 0.9825 | 0.9673 | 0.9761 | 0.9921 | 0.9986 | 0.9973 | 270 |
| PNEUMONIA | 0.9630 | 0.9903 | 0.9811 | 0.9720 | 0.9807 | 0.9968 | 0.9904 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=266 FP=9 TN=505 FN=4
- **PNEUMONIA**: TP=260 FP=5 TN=509 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 266 | 4 |
| PNEUMONIA | 1 | 9 | 260 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9673    0.9852    0.9761       270
   PNEUMONIA     0.9811    0.9630    0.9720       270

    accuracy                         0.9809       784
   macro avg     0.9814    0.9813    0.9813       784
weighted avg     0.9810    0.9809    0.9809       784

```