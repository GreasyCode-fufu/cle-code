# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6/mbv3_fz16_dce4m_seed44/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9876 |
| Accuracy | 0.9873 |
| Macro-AUROC (OvR) | 0.9990 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9873 |
| Balanced Acc | 0.9874 |
| Macro-Sensitivity | 0.9874 |
| Macro-Specificity | 0.9935 |
| Macro-Precision | 0.9877 |
| AUC-PR (Macro) | 0.9982 |
| Cohen's Kappa | 0.9809 |
| MCC | 0.9809 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3963.3 (0.25 ms/张) |
| End-to-End FPS | 16.6 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9852 | 0.9903 | 0.9816 | 0.9834 | 0.9922 | 0.9987 | 0.9972 | 271 |
| PNEUMONIA | 0.9852 | 0.9903 | 0.9815 | 0.9834 | 0.9922 | 0.9982 | 0.9975 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=267 FP=5 TN=509 FN=4
- **PNEUMONIA**: TP=266 FP=5 TN=510 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 267 | 4 |
| PNEUMONIA | 0 | 4 | 266 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9816    0.9852    0.9834       271
   PNEUMONIA     0.9815    0.9852    0.9834       270

    accuracy                         0.9873       785
   macro avg     0.9877    0.9874    0.9876       785
weighted avg     0.9873    0.9873    0.9873       785

```