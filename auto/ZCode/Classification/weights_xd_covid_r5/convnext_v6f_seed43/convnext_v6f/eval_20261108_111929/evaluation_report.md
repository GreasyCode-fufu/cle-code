# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r5/convnext_v6f_seed43/convnext_v6f/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9849 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9848 |
| Macro-Sensitivity | 0.9848 |
| Macro-Specificity | 0.9923 |
| Macro-Precision | 0.9851 |
| AUC-PR (Macro) | 0.9987 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9771 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 693.1 (1.44 ms/张) |
| End-to-End FPS | 392.0 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9877 | 0.9982 | 0.9959 | 0.9918 | 0.9945 | 0.9999 | 0.9998 | 244 |
| NORMAL | 0.9889 | 0.9883 | 0.9781 | 0.9835 | 0.9941 | 0.9995 | 0.9990 | 271 |
| PNEUMONIA | 0.9778 | 0.9903 | 0.9814 | 0.9796 | 0.9884 | 0.9987 | 0.9974 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=241 FP=1 TN=540 FN=3
- **NORMAL**: TP=268 FP=6 TN=508 FN=3
- **PNEUMONIA**: TP=264 FP=5 TN=510 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 241 | 1 | 2 |
| NORMAL | 0 | 268 | 3 |
| PNEUMONIA | 1 | 5 | 264 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9877    0.9918       244
      NORMAL     0.9781    0.9889    0.9835       271
   PNEUMONIA     0.9814    0.9778    0.9796       270

    accuracy                         0.9847       785
   macro avg     0.9851    0.9848    0.9849       785
weighted avg     0.9848    0.9847    0.9847       785

```