# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/pvt_tiny_seed43/pvt_tiny/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9864 |
| Accuracy | 0.9860 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9860 |
| Balanced Acc | 0.9863 |
| Macro-Sensitivity | 0.9863 |
| Macro-Specificity | 0.9929 |
| Macro-Precision | 0.9865 |
| AUC-PR (Macro) | 0.9986 |
| Cohen's Kappa | 0.9790 |
| MCC | 0.9790 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2925.9 (0.34 ms/张) |
| End-to-End FPS | 712.5 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 1.0000 | 1.0000 | 0.9979 | 0.9982 | 1.0000 | 1.0000 | 244 |
| NORMAL | 0.9779 | 0.9922 | 0.9851 | 0.9815 | 0.9884 | 0.9990 | 0.9980 | 271 |
| PNEUMONIA | 0.9852 | 0.9864 | 0.9744 | 0.9797 | 0.9922 | 0.9988 | 0.9979 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=0 TN=541 FN=1
- **NORMAL**: TP=265 FP=4 TN=510 FN=6
- **PNEUMONIA**: TP=266 FP=7 TN=508 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 265 | 6 |
| PNEUMONIA | 0 | 4 | 266 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9959    0.9979       244
      NORMAL     0.9851    0.9779    0.9815       271
   PNEUMONIA     0.9744    0.9852    0.9797       270

    accuracy                         0.9860       785
   macro avg     0.9865    0.9863    0.9864       785
weighted avg     0.9860    0.9860    0.9860       785

```