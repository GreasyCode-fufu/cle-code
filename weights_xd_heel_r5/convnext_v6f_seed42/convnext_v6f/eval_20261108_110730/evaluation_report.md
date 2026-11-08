# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r5/convnext_v6f_seed42/convnext_v6f/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9022 |
| Accuracy | 0.8954 |
| Macro-AUROC (OvR) | 0.9822 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8949 |
| Balanced Acc | 0.9054 |
| Macro-Sensitivity | 0.9054 |
| Macro-Specificity | 0.9428 |
| Macro-Precision | 0.8997 |
| AUC-PR (Macro) | 0.9740 |
| Cohen's Kappa | 0.8348 |
| MCC | 0.8350 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 598.2 (1.67 ms/张) |
| End-to-End FPS | 240.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8426 | 0.9419 | 0.8783 | 0.8601 | 0.9233 | 0.9726 | 0.9552 | 197 |
| normal | 0.8986 | 0.9054 | 0.8921 | 0.8953 | 0.9111 | 0.9751 | 0.9710 | 276 |
| sever | 0.9750 | 0.9810 | 0.9286 | 0.9512 | 0.9936 | 0.9989 | 0.9959 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=166 FP=23 TN=373 FN=31
- **normal**: TP=248 FP=30 TN=287 FN=28
- **sever**: TP=117 FP=9 TN=464 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 166 | 27 | 4 |
| normal | 23 | 248 | 5 |
| sever | 0 | 3 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8783    0.8426    0.8601       197
      normal     0.8921    0.8986    0.8953       276
       sever     0.9286    0.9750    0.9512       120

    accuracy                         0.8954       593
   macro avg     0.8997    0.9054    0.9022       593
weighted avg     0.8949    0.8954    0.8949       593

```