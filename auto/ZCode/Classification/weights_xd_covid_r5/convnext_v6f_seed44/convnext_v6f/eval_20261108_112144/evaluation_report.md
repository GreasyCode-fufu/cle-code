# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r5/convnext_v6f_seed44/convnext_v6f/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9851 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9995 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9849 |
| Macro-Sensitivity | 0.9849 |
| Macro-Specificity | 0.9922 |
| Macro-Precision | 0.9855 |
| AUC-PR (Macro) | 0.9991 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9772 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 855.7 (1.17 ms/张) |
| End-to-End FPS | 445.6 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9963 | 0.9825 | 0.9677 | 0.9818 | 0.9980 | 0.9995 | 0.9989 | 271 |
| PNEUMONIA | 0.9667 | 0.9942 | 0.9886 | 0.9775 | 0.9827 | 0.9991 | 0.9984 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=270 FP=9 TN=505 FN=1
- **PNEUMONIA**: TP=261 FP=3 TN=512 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 0 | 2 |
| NORMAL | 0 | 270 | 1 |
| PNEUMONIA | 0 | 9 | 261 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9677    0.9963    0.9818       271
   PNEUMONIA     0.9886    0.9667    0.9775       270

    accuracy                         0.9847       785
   macro avg     0.9855    0.9849    0.9851       785
weighted avg     0.9850    0.9847    0.9847       785

```