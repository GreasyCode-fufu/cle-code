# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/pvt_tiny_seed42/pvt_tiny/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9863 |
| Accuracy | 0.9860 |
| Macro-AUROC (OvR) | 0.9985 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9860 |
| Balanced Acc | 0.9862 |
| Macro-Sensitivity | 0.9862 |
| Macro-Specificity | 0.9929 |
| Macro-Precision | 0.9865 |
| AUC-PR (Macro) | 0.9970 |
| Cohen's Kappa | 0.9790 |
| MCC | 0.9790 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2918.5 (0.34 ms/张) |
| End-to-End FPS | 712.4 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 0.9999 | 0.9998 | 244 |
| NORMAL | 0.9889 | 0.9864 | 0.9745 | 0.9817 | 0.9941 | 0.9979 | 0.9957 | 271 |
| PNEUMONIA | 0.9778 | 0.9922 | 0.9851 | 0.9814 | 0.9884 | 0.9976 | 0.9956 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=268 FP=7 TN=507 FN=3
- **PNEUMONIA**: TP=264 FP=4 TN=511 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 268 | 3 |
| PNEUMONIA | 0 | 6 | 264 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9745    0.9889    0.9817       271
   PNEUMONIA     0.9851    0.9778    0.9814       270

    accuracy                         0.9860       785
   macro avg     0.9865    0.9862    0.9863       785
weighted avg     0.9861    0.9860    0.9860       785

```