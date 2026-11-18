# convnext_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_dsd_seed44/convnext_dsd/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9912 |
| Accuracy | 0.9911 |
| Macro-AUROC (OvR) | 0.9996 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9911 |
| Balanced Acc | 0.9911 |
| Macro-Sensitivity | 0.9911 |
| Macro-Specificity | 0.9955 |
| Macro-Precision | 0.9914 |
| AUC-PR (Macro) | 0.9992 |
| Cohen's Kappa | 0.9866 |
| MCC | 0.9866 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 842.8 (1.19 ms/张) |
| End-to-End FPS | 440.0 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 0.9999 | 0.9998 | 244 |
| NORMAL | 0.9926 | 0.9922 | 0.9853 | 0.9890 | 0.9961 | 0.9994 | 0.9989 | 271 |
| PNEUMONIA | 0.9889 | 0.9942 | 0.9889 | 0.9889 | 0.9942 | 0.9993 | 0.9988 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=269 FP=4 TN=510 FN=2
- **PNEUMONIA**: TP=267 FP=3 TN=512 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 269 | 2 |
| PNEUMONIA | 0 | 3 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9853    0.9926    0.9890       271
   PNEUMONIA     0.9889    0.9889    0.9889       270

    accuracy                         0.9911       785
   macro avg     0.9914    0.9911    0.9912       785
weighted avg     0.9911    0.9911    0.9911       785

```