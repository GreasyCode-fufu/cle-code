# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/resnet34_seed43/resnet34/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9862 |
| Accuracy | 0.9860 |
| Macro-AUROC (OvR) | 0.9973 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9860 |
| Balanced Acc | 0.9863 |
| Macro-Sensitivity | 0.9863 |
| Macro-Specificity | 0.9929 |
| Macro-Precision | 0.9861 |
| AUC-PR (Macro) | 0.9951 |
| Cohen's Kappa | 0.9789 |
| MCC | 0.9789 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3825.1 (0.26 ms/张) |
| End-to-End FPS | 750.5 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9963 | 0.9918 | 0.9939 | 0.9981 | 0.9962 | 0.9970 | 244 |
| NORMAL | 0.9852 | 0.9922 | 0.9852 | 0.9852 | 0.9922 | 0.9990 | 0.9984 | 270 |
| PNEUMONIA | 0.9778 | 0.9903 | 0.9814 | 0.9796 | 0.9883 | 0.9967 | 0.9898 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=2 TN=538 FN=1
- **NORMAL**: TP=266 FP=4 TN=510 FN=4
- **PNEUMONIA**: TP=264 FP=5 TN=509 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 266 | 4 |
| PNEUMONIA | 2 | 4 | 264 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9918    0.9959    0.9939       244
      NORMAL     0.9852    0.9852    0.9852       270
   PNEUMONIA     0.9814    0.9778    0.9796       270

    accuracy                         0.9860       784
   macro avg     0.9861    0.9863    0.9862       784
weighted avg     0.9860    0.9860    0.9860       784

```