# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r3/convnext_tiny_seed43/convnext_tiny/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9814 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9972 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9813 |
| Macro-Sensitivity | 0.9813 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9814 |
| AUC-PR (Macro) | 0.9946 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9713 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 968.3 (1.03 ms/张) |
| End-to-End FPS | 472.2 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9962 | 0.9971 | 244 |
| NORMAL | 0.9704 | 0.9903 | 0.9813 | 0.9758 | 0.9845 | 0.9987 | 0.9978 | 270 |
| PNEUMONIA | 0.9778 | 0.9825 | 0.9670 | 0.9724 | 0.9883 | 0.9966 | 0.9887 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=262 FP=5 TN=509 FN=8
- **PNEUMONIA**: TP=264 FP=9 TN=505 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 262 | 8 |
| PNEUMONIA | 1 | 5 | 264 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9813    0.9704    0.9758       270
   PNEUMONIA     0.9670    0.9778    0.9724       270

    accuracy                         0.9809       784
   macro avg     0.9814    0.9813    0.9814       784
weighted avg     0.9809    0.9809    0.9809       784

```