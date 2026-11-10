# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/jumbo_small_seed43/jumbo_small/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9800 |
| Accuracy | 0.9796 |
| Macro-AUROC (OvR) | 0.9957 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9796 |
| Balanced Acc | 0.9800 |
| Macro-Sensitivity | 0.9800 |
| Macro-Specificity | 0.9897 |
| Macro-Precision | 0.9801 |
| AUC-PR (Macro) | 0.9931 |
| Cohen's Kappa | 0.9694 |
| MCC | 0.9694 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 1170.7 (0.85 ms/张) |
| End-to-End FPS | 514.9 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9963 | 0.9918 | 0.9918 | 0.9963 | 0.9959 | 0.9957 | 244 |
| NORMAL | 0.9852 | 0.9825 | 0.9673 | 0.9761 | 0.9921 | 0.9974 | 0.9954 | 270 |
| PNEUMONIA | 0.9630 | 0.9903 | 0.9811 | 0.9720 | 0.9807 | 0.9937 | 0.9884 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=2 TN=538 FN=2
- **NORMAL**: TP=266 FP=9 TN=505 FN=4
- **PNEUMONIA**: TP=260 FP=5 TN=509 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 266 | 4 |
| PNEUMONIA | 2 | 8 | 260 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9918    0.9918    0.9918       244
      NORMAL     0.9673    0.9852    0.9761       270
   PNEUMONIA     0.9811    0.9630    0.9720       270

    accuracy                         0.9796       784
   macro avg     0.9801    0.9800    0.9800       784
weighted avg     0.9797    0.9796    0.9796       784

```