# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_ep/convnext_svd_seed44/convnext_svd/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9825 |
| Accuracy | 0.9821 |
| Macro-AUROC (OvR) | 0.9974 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9821 |
| Balanced Acc | 0.9825 |
| Macro-Sensitivity | 0.9825 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9827 |
| AUC-PR (Macro) | 0.9956 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9733 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 1845.1 (0.54 ms/张) |
| End-to-End FPS | 623.2 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9981 | 0.9959 | 0.9938 | 0.9963 | 0.9963 | 0.9969 | 244 |
| NORMAL | 0.9889 | 0.9825 | 0.9674 | 0.9780 | 0.9941 | 0.9988 | 0.9980 | 270 |
| PNEUMONIA | 0.9667 | 0.9922 | 0.9849 | 0.9757 | 0.9827 | 0.9970 | 0.9918 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=1 TN=539 FN=2
- **NORMAL**: TP=267 FP=9 TN=505 FN=3
- **PNEUMONIA**: TP=261 FP=4 TN=510 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 267 | 3 |
| PNEUMONIA | 1 | 8 | 261 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9918    0.9938       244
      NORMAL     0.9674    0.9889    0.9780       270
   PNEUMONIA     0.9849    0.9667    0.9757       270

    accuracy                         0.9821       784
   macro avg     0.9827    0.9825    0.9825       784
weighted avg     0.9823    0.9821    0.9821       784

```