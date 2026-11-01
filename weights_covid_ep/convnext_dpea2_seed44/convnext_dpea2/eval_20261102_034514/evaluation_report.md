# convnext_dpea2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_dpea2_seed44/convnext_dpea2/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9850 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9851 |
| Macro-Sensitivity | 0.9851 |
| Macro-Specificity | 0.9922 |
| Macro-Precision | 0.9851 |
| AUC-PR (Macro) | 0.9941 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9770 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 28.12M |
| Pure-Inference FPS | 355.5 (2.81 ms/张) |
| End-to-End FPS | 254.5 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9956 | 0.9947 | 244 |
| NORMAL | 0.9852 | 0.9883 | 0.9779 | 0.9815 | 0.9922 | 0.9991 | 0.9985 | 270 |
| PNEUMONIA | 0.9741 | 0.9903 | 0.9813 | 0.9777 | 0.9864 | 0.9947 | 0.9890 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=266 FP=6 TN=508 FN=4
- **PNEUMONIA**: TP=263 FP=5 TN=509 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 266 | 4 |
| PNEUMONIA | 1 | 6 | 263 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9779    0.9852    0.9815       270
   PNEUMONIA     0.9813    0.9741    0.9777       270

    accuracy                         0.9847       784
   macro avg     0.9851    0.9851    0.9850       784
weighted avg     0.9847    0.9847    0.9847       784

```