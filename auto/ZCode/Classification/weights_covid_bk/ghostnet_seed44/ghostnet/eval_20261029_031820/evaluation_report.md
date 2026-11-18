# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/ghostnet_seed44/ghostnet/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9814 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9984 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9813 |
| Macro-Sensitivity | 0.9813 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9818 |
| AUC-PR (Macro) | 0.9972 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9714 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 5026.6 (0.20 ms/张) |
| End-to-End FPS | 784.0 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 0.9999 | 0.9999 | 244 |
| NORMAL | 0.9668 | 0.9922 | 0.9850 | 0.9758 | 0.9827 | 0.9980 | 0.9959 | 271 |
| PNEUMONIA | 0.9852 | 0.9786 | 0.9603 | 0.9726 | 0.9921 | 0.9973 | 0.9958 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=262 FP=4 TN=510 FN=9
- **PNEUMONIA**: TP=266 FP=11 TN=504 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 0 | 2 |
| NORMAL | 0 | 262 | 9 |
| PNEUMONIA | 0 | 4 | 266 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9850    0.9668    0.9758       271
   PNEUMONIA     0.9603    0.9852    0.9726       270

    accuracy                         0.9809       785
   macro avg     0.9818    0.9813    0.9814       785
weighted avg     0.9812    0.9809    0.9809       785

```