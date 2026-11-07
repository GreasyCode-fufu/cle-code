# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r3/convnext_v4e_seed42/convnext_v4e/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9874 |
| Accuracy | 0.9872 |
| Macro-AUROC (OvR) | 0.9970 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9872 |
| Balanced Acc | 0.9875 |
| Macro-Sensitivity | 0.9875 |
| Macro-Specificity | 0.9936 |
| Macro-Precision | 0.9874 |
| AUC-PR (Macro) | 0.9928 |
| Cohen's Kappa | 0.9808 |
| MCC | 0.9809 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 417.4 (2.40 ms/张) |
| End-to-End FPS | 286.8 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9963 | 0.9918 | 0.9939 | 0.9981 | 0.9959 | 0.9971 | 244 |
| NORMAL | 0.9889 | 0.9922 | 0.9852 | 0.9871 | 0.9942 | 0.9987 | 0.9977 | 270 |
| PNEUMONIA | 0.9778 | 0.9922 | 0.9851 | 0.9814 | 0.9884 | 0.9964 | 0.9838 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=2 TN=538 FN=1
- **NORMAL**: TP=267 FP=4 TN=510 FN=3
- **PNEUMONIA**: TP=264 FP=4 TN=510 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 267 | 3 |
| PNEUMONIA | 2 | 4 | 264 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9918    0.9959    0.9939       244
      NORMAL     0.9852    0.9889    0.9871       270
   PNEUMONIA     0.9851    0.9778    0.9814       270

    accuracy                         0.9872       784
   macro avg     0.9874    0.9875    0.9874       784
weighted avg     0.9872    0.9872    0.9872       784

```