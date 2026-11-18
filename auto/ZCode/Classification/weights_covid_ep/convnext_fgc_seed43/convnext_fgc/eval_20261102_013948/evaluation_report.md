# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_fgc_seed43/convnext_fgc/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9912 |
| Accuracy | 0.9911 |
| Macro-AUROC (OvR) | 0.9997 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9911 |
| Balanced Acc | 0.9912 |
| Macro-Sensitivity | 0.9912 |
| Macro-Specificity | 0.9955 |
| Macro-Precision | 0.9912 |
| AUC-PR (Macro) | 0.9994 |
| Cohen's Kappa | 0.9866 |
| MCC | 0.9866 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 370.3 (2.70 ms/张) |
| End-to-End FPS | 263.1 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9982 | 0.9959 | 0.9959 | 0.9982 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9889 | 0.9961 | 0.9926 | 0.9908 | 0.9942 | 0.9998 | 0.9996 | 271 |
| PNEUMONIA | 0.9889 | 0.9922 | 0.9852 | 0.9871 | 0.9942 | 0.9993 | 0.9987 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=540 FN=1
- **NORMAL**: TP=268 FP=2 TN=512 FN=3
- **PNEUMONIA**: TP=267 FP=4 TN=511 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 268 | 3 |
| PNEUMONIA | 1 | 2 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9926    0.9889    0.9908       271
   PNEUMONIA     0.9852    0.9889    0.9871       270

    accuracy                         0.9911       785
   macro avg     0.9912    0.9912    0.9912       785
weighted avg     0.9911    0.9911    0.9911       785

```