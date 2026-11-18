# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r4/convnext_v2c_seed42/convnext_v2c/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9814 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9988 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9812 |
| Macro-Sensitivity | 0.9812 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9816 |
| AUC-PR (Macro) | 0.9978 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9713 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 841.7 (1.19 ms/张) |
| End-to-End FPS | 440.7 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 0.9999 | 0.9999 | 244 |
| NORMAL | 0.9779 | 0.9844 | 0.9707 | 0.9743 | 0.9883 | 0.9984 | 0.9969 | 271 |
| PNEUMONIA | 0.9741 | 0.9864 | 0.9741 | 0.9741 | 0.9864 | 0.9980 | 0.9968 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=265 FP=8 TN=506 FN=6
- **PNEUMONIA**: TP=263 FP=7 TN=508 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 265 | 6 |
| PNEUMONIA | 0 | 7 | 263 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9707    0.9779    0.9743       271
   PNEUMONIA     0.9741    0.9741    0.9741       270

    accuracy                         0.9809       785
   macro avg     0.9816    0.9812    0.9814       785
weighted avg     0.9810    0.9809    0.9809       785

```