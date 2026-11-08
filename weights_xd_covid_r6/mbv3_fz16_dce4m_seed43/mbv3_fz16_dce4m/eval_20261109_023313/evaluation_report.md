# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6/mbv3_fz16_dce4m_seed43/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9852 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9851 |
| Macro-Sensitivity | 0.9851 |
| Macro-Specificity | 0.9922 |
| Macro-Precision | 0.9852 |
| AUC-PR (Macro) | 0.9987 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9770 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 2953.9 (0.34 ms/张) |
| End-to-End FPS | 15.3 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 1.0000 | 1.0000 | 0.9979 | 0.9982 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9779 | 0.9903 | 0.9815 | 0.9797 | 0.9883 | 0.9990 | 0.9981 | 271 |
| PNEUMONIA | 0.9815 | 0.9864 | 0.9743 | 0.9779 | 0.9903 | 0.9989 | 0.9981 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=0 TN=541 FN=1
- **NORMAL**: TP=265 FP=5 TN=509 FN=6
- **PNEUMONIA**: TP=265 FP=7 TN=508 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 265 | 6 |
| PNEUMONIA | 0 | 5 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9959    0.9979       244
      NORMAL     0.9815    0.9779    0.9797       271
   PNEUMONIA     0.9743    0.9815    0.9779       270

    accuracy                         0.9847       785
   macro avg     0.9852    0.9851    0.9852       785
weighted avg     0.9848    0.9847    0.9847       785

```