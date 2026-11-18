# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/fast_t0_seed43/fast_t0/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9851 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9991 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9850 |
| Macro-Sensitivity | 0.9850 |
| Macro-Specificity | 0.9922 |
| Macro-Precision | 0.9854 |
| AUC-PR (Macro) | 0.9983 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9771 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 9851.2 (0.10 ms/张) |
| End-to-End FPS | 843.7 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 0.9999 | 0.9998 | 244 |
| NORMAL | 0.9742 | 0.9942 | 0.9888 | 0.9814 | 0.9865 | 0.9988 | 0.9977 | 271 |
| PNEUMONIA | 0.9889 | 0.9825 | 0.9674 | 0.9780 | 0.9941 | 0.9985 | 0.9973 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=264 FP=3 TN=511 FN=7
- **PNEUMONIA**: TP=267 FP=9 TN=506 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 0 | 2 |
| NORMAL | 0 | 264 | 7 |
| PNEUMONIA | 0 | 3 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9888    0.9742    0.9814       271
   PNEUMONIA     0.9674    0.9889    0.9780       270

    accuracy                         0.9847       785
   macro avg     0.9854    0.9850    0.9851       785
weighted avg     0.9849    0.9847    0.9847       785

```