# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/repvit_m1_seed43/repvit_m1/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9814 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9985 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9814 |
| Macro-Sensitivity | 0.9814 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9814 |
| AUC-PR (Macro) | 0.9973 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9713 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3911.2 (0.26 ms/张) |
| End-to-End FPS | 746.0 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9982 | 0.9959 | 0.9959 | 0.9982 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9668 | 0.9903 | 0.9813 | 0.9740 | 0.9826 | 0.9981 | 0.9968 | 271 |
| PNEUMONIA | 0.9815 | 0.9825 | 0.9672 | 0.9743 | 0.9902 | 0.9974 | 0.9952 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=540 FN=1
- **NORMAL**: TP=262 FP=5 TN=509 FN=9
- **PNEUMONIA**: TP=265 FP=9 TN=506 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 1 | 0 |
| NORMAL | 0 | 262 | 9 |
| PNEUMONIA | 1 | 4 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9813    0.9668    0.9740       271
   PNEUMONIA     0.9672    0.9815    0.9743       270

    accuracy                         0.9809       785
   macro avg     0.9814    0.9814    0.9814       785
weighted avg     0.9810    0.9809    0.9809       785

```