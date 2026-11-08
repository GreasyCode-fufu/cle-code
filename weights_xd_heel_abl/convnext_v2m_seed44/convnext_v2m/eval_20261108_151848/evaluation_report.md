# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_abl/convnext_v2m_seed44/convnext_v2m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9034 |
| Accuracy | 0.8954 |
| Macro-AUROC (OvR) | 0.9798 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8947 |
| Balanced Acc | 0.9081 |
| Macro-Sensitivity | 0.9081 |
| Macro-Specificity | 0.9428 |
| Macro-Precision | 0.8997 |
| AUC-PR (Macro) | 0.9696 |
| Cohen's Kappa | 0.8350 |
| MCC | 0.8353 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 837.1 (1.19 ms/张) |
| End-to-End FPS | 268.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8376 | 0.9419 | 0.8777 | 0.8571 | 0.9210 | 0.9680 | 0.9491 | 197 |
| normal | 0.8949 | 0.9054 | 0.8917 | 0.8933 | 0.9082 | 0.9722 | 0.9627 | 276 |
| sever | 0.9917 | 0.9810 | 0.9297 | 0.9597 | 0.9978 | 0.9993 | 0.9970 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=165 FP=23 TN=373 FN=32
- **normal**: TP=247 FP=30 TN=287 FN=29
- **sever**: TP=119 FP=9 TN=464 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 165 | 29 | 3 |
| normal | 23 | 247 | 6 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8777    0.8376    0.8571       197
      normal     0.8917    0.8949    0.8933       276
       sever     0.9297    0.9917    0.9597       120

    accuracy                         0.8954       593
   macro avg     0.8997    0.9081    0.9034       593
weighted avg     0.8947    0.8954    0.8947       593

```