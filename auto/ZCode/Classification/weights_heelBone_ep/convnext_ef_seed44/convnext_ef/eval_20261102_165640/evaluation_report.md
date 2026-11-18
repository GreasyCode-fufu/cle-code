# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_ef_seed44/convnext_ef/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8903 |
| Accuracy | 0.8803 |
| Macro-AUROC (OvR) | 0.9787 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8803 |
| Balanced Acc | 0.8984 |
| Macro-Sensitivity | 0.8984 |
| Macro-Specificity | 0.9371 |
| Macro-Precision | 0.8848 |
| AUC-PR (Macro) | 0.9695 |
| Cohen's Kappa | 0.8126 |
| MCC | 0.8143 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 930.2 (1.07 ms/张) |
| End-to-End FPS | 281.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8832 | 0.8965 | 0.8093 | 0.8447 | 0.9392 | 0.9666 | 0.9480 | 197 |
| normal | 0.8370 | 0.9338 | 0.9167 | 0.8750 | 0.8680 | 0.9711 | 0.9665 | 276 |
| sever | 0.9750 | 0.9810 | 0.9286 | 0.9512 | 0.9936 | 0.9984 | 0.9940 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=174 FP=41 TN=355 FN=23
- **normal**: TP=231 FP=21 TN=296 FN=45
- **sever**: TP=117 FP=9 TN=464 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 174 | 19 | 4 |
| normal | 40 | 231 | 5 |
| sever | 1 | 2 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8093    0.8832    0.8447       197
      normal     0.9167    0.8370    0.8750       276
       sever     0.9286    0.9750    0.9512       120

    accuracy                         0.8803       593
   macro avg     0.8848    0.8984    0.8903       593
weighted avg     0.8834    0.8803    0.8803       593

```