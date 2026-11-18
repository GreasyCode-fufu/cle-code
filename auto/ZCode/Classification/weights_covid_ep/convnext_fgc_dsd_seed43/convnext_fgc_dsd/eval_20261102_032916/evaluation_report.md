# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_fgc_dsd_seed43/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9837 |
| Accuracy | 0.9834 |
| Macro-AUROC (OvR) | 0.9991 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9835 |
| Balanced Acc | 0.9834 |
| Macro-Sensitivity | 0.9834 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9841 |
| AUC-PR (Macro) | 0.9982 |
| Cohen's Kappa | 0.9751 |
| MCC | 0.9752 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 380.2 (2.63 ms/张) |
| End-to-End FPS | 266.2 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9836 | 1.0000 | 1.0000 | 0.9917 | 0.9927 | 0.9997 | 0.9994 | 244 |
| NORMAL | 0.9815 | 0.9903 | 0.9815 | 0.9815 | 0.9903 | 0.9990 | 0.9980 | 271 |
| PNEUMONIA | 0.9852 | 0.9845 | 0.9708 | 0.9779 | 0.9922 | 0.9985 | 0.9972 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=240 FP=0 TN=541 FN=4
- **NORMAL**: TP=266 FP=5 TN=509 FN=5
- **PNEUMONIA**: TP=266 FP=8 TN=507 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 240 | 1 | 3 |
| NORMAL | 0 | 266 | 5 |
| PNEUMONIA | 0 | 4 | 266 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9836    0.9917       244
      NORMAL     0.9815    0.9815    0.9815       271
   PNEUMONIA     0.9708    0.9852    0.9779       270

    accuracy                         0.9834       785
   macro avg     0.9841    0.9834    0.9837       785
weighted avg     0.9836    0.9834    0.9835       785

```