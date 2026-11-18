# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mobilenet_v3_large_seed44/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8249 |
| Accuracy | 0.8179 |
| Macro-AUROC (OvR) | 0.9482 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8163 |
| Balanced Acc | 0.8462 |
| Macro-Sensitivity | 0.8462 |
| Macro-Specificity | 0.9063 |
| Macro-Precision | 0.8122 |
| AUC-PR (Macro) | 0.9249 |
| Cohen's Kappa | 0.7176 |
| MCC | 0.7204 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 3080.2 (0.32 ms/张) |
| End-to-End FPS | 159.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7716 | 0.8864 | 0.7716 | 0.7716 | 0.8864 | 0.9162 | 0.8714 | 197 |
| normal | 0.7754 | 0.8959 | 0.8664 | 0.8184 | 0.8208 | 0.9321 | 0.9165 | 276 |
| sever | 0.9917 | 0.9366 | 0.7987 | 0.8848 | 0.9977 | 0.9964 | 0.9870 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=152 FP=45 TN=351 FN=45
- **normal**: TP=214 FP=33 TN=284 FN=62
- **sever**: TP=119 FP=30 TN=443 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 152 | 32 | 13 |
| normal | 45 | 214 | 17 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7716    0.7716    0.7716       197
      normal     0.8664    0.7754    0.8184       276
       sever     0.7987    0.9917    0.8848       120

    accuracy                         0.8179       593
   macro avg     0.8122    0.8462    0.8249       593
weighted avg     0.8212    0.8179    0.8163       593

```