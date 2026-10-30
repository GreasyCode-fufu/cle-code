# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/resnet34_seed43/resnet34/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9826 |
| Accuracy | 0.9822 |
| Macro-AUROC (OvR) | 0.9992 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9822 |
| Balanced Acc | 0.9825 |
| Macro-Sensitivity | 0.9825 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9827 |
| AUC-PR (Macro) | 0.9985 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3846.4 (0.26 ms/张) |
| End-to-End FPS | 755.4 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9982 | 0.9959 | 0.9938 | 0.9963 | 0.9999 | 0.9998 | 244 |
| NORMAL | 0.9742 | 0.9903 | 0.9814 | 0.9778 | 0.9864 | 0.9988 | 0.9976 | 271 |
| PNEUMONIA | 0.9815 | 0.9845 | 0.9707 | 0.9761 | 0.9902 | 0.9989 | 0.9981 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=1 TN=540 FN=2
- **NORMAL**: TP=264 FP=5 TN=509 FN=7
- **PNEUMONIA**: TP=265 FP=8 TN=507 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 264 | 7 |
| PNEUMONIA | 1 | 4 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9918    0.9938       244
      NORMAL     0.9814    0.9742    0.9778       271
   PNEUMONIA     0.9707    0.9815    0.9761       270

    accuracy                         0.9822       785
   macro avg     0.9827    0.9825    0.9826       785
weighted avg     0.9822    0.9822    0.9822       785

```