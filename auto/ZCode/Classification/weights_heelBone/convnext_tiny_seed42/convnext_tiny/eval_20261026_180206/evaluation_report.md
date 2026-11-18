# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/convnext_tiny_seed42/convnext_tiny/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9040 |
| Accuracy | 0.8954 |
| Macro-AUROC (OvR) | 0.9786 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8950 |
| Balanced Acc | 0.9019 |
| Macro-Sensitivity | 0.9019 |
| Macro-Specificity | 0.9410 |
| Macro-Precision | 0.9069 |
| AUC-PR (Macro) | 0.9683 |
| Cohen's Kappa | 0.8339 |
| MCC | 0.8344 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 605.4 (1.65 ms/张) |
| End-to-End FPS | 242.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8223 | 0.9419 | 0.8757 | 0.8482 | 0.9142 | 0.9662 | 0.9465 | 197 |
| normal | 0.9167 | 0.8896 | 0.8785 | 0.8972 | 0.9246 | 0.9712 | 0.9636 | 276 |
| sever | 0.9667 | 0.9915 | 0.9667 | 0.9667 | 0.9915 | 0.9984 | 0.9947 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=162 FP=23 TN=373 FN=35
- **normal**: TP=253 FP=35 TN=282 FN=23
- **sever**: TP=116 FP=4 TN=469 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 162 | 33 | 2 |
| normal | 21 | 253 | 2 |
| sever | 2 | 2 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8757    0.8223    0.8482       197
      normal     0.8785    0.9167    0.8972       276
       sever     0.9667    0.9667    0.9667       120

    accuracy                         0.8954       593
   macro avg     0.9069    0.9019    0.9040       593
weighted avg     0.8954    0.8954    0.8950       593

```