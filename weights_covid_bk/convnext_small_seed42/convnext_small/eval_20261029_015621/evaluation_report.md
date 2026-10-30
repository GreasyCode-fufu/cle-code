# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/convnext_small_seed42/convnext_small/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9875 |
| Accuracy | 0.9872 |
| Macro-AUROC (OvR) | 0.9979 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9872 |
| Balanced Acc | 0.9875 |
| Macro-Sensitivity | 0.9875 |
| Macro-Specificity | 0.9935 |
| Macro-Precision | 0.9875 |
| AUC-PR (Macro) | 0.9961 |
| Cohen's Kappa | 0.9808 |
| MCC | 0.9808 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 1059.8 (0.94 ms/张) |
| End-to-End FPS | 498.4 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9965 | 0.9973 | 244 |
| NORMAL | 0.9852 | 0.9922 | 0.9852 | 0.9852 | 0.9922 | 0.9993 | 0.9987 | 270 |
| PNEUMONIA | 0.9815 | 0.9903 | 0.9815 | 0.9815 | 0.9903 | 0.9978 | 0.9924 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=266 FP=4 TN=510 FN=4
- **PNEUMONIA**: TP=265 FP=5 TN=509 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 266 | 4 |
| PNEUMONIA | 1 | 4 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9852    0.9852    0.9852       270
   PNEUMONIA     0.9815    0.9815    0.9815       270

    accuracy                         0.9872       784
   macro avg     0.9875    0.9875    0.9875       784
weighted avg     0.9872    0.9872    0.9872       784

```