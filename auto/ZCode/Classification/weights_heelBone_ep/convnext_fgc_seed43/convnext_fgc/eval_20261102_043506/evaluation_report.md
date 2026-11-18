# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_fgc_seed43/convnext_fgc/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8909 |
| Accuracy | 0.8870 |
| Macro-AUROC (OvR) | 0.9721 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8857 |
| Balanced Acc | 0.8981 |
| Macro-Sensitivity | 0.8981 |
| Macro-Specificity | 0.9384 |
| Macro-Precision | 0.8891 |
| AUC-PR (Macro) | 0.9623 |
| Cohen's Kappa | 0.8216 |
| MCC | 0.8235 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 903.5 (1.11 ms/张) |
| End-to-End FPS | 281.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7970 | 0.9646 | 0.9181 | 0.8533 | 0.9052 | 0.9516 | 0.9330 | 197 |
| normal | 0.9058 | 0.8864 | 0.8741 | 0.8897 | 0.9153 | 0.9659 | 0.9583 | 276 |
| sever | 0.9917 | 0.9641 | 0.8750 | 0.9297 | 0.9978 | 0.9988 | 0.9956 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=157 FP=14 TN=382 FN=40
- **normal**: TP=250 FP=36 TN=281 FN=26
- **sever**: TP=119 FP=17 TN=456 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 157 | 35 | 5 |
| normal | 14 | 250 | 12 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.9181    0.7970    0.8533       197
      normal     0.8741    0.9058    0.8897       276
       sever     0.8750    0.9917    0.9297       120

    accuracy                         0.8870       593
   macro avg     0.8891    0.8981    0.8909       593
weighted avg     0.8889    0.8870    0.8857       593

```