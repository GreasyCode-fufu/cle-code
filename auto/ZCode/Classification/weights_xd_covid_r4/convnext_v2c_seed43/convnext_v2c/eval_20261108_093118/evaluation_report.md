# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r4/convnext_v2c_seed43/convnext_v2c/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9789 |
| Accuracy | 0.9783 |
| Macro-AUROC (OvR) | 0.9995 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9784 |
| Balanced Acc | 0.9788 |
| Macro-Sensitivity | 0.9788 |
| Macro-Specificity | 0.9890 |
| Macro-Precision | 0.9795 |
| AUC-PR (Macro) | 0.9991 |
| Cohen's Kappa | 0.9675 |
| MCC | 0.9678 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1682.2 (0.59 ms/张) |
| End-to-End FPS | 606.4 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9982 | 0.9959 | 0.9938 | 0.9963 | 0.9999 | 0.9999 | 244 |
| NORMAL | 0.9520 | 0.9961 | 0.9923 | 0.9718 | 0.9752 | 0.9994 | 0.9989 | 271 |
| PNEUMONIA | 0.9926 | 0.9728 | 0.9504 | 0.9710 | 0.9960 | 0.9992 | 0.9986 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=1 TN=540 FN=2
- **NORMAL**: TP=258 FP=2 TN=512 FN=13
- **PNEUMONIA**: TP=268 FP=14 TN=501 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 258 | 13 |
| PNEUMONIA | 1 | 1 | 268 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9918    0.9938       244
      NORMAL     0.9923    0.9520    0.9718       271
   PNEUMONIA     0.9504    0.9926    0.9710       270

    accuracy                         0.9783       785
   macro avg     0.9795    0.9788    0.9789       785
weighted avg     0.9790    0.9783    0.9784       785

```