# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/convnext_v2bc_seed43/convnext_v2bc/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8636 |
| Accuracy | 0.8600 |
| Macro-AUROC (OvR) | 0.9744 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8593 |
| Balanced Acc | 0.8877 |
| Macro-Sensitivity | 0.8877 |
| Macro-Specificity | 0.9311 |
| Macro-Precision | 0.8516 |
| AUC-PR (Macro) | 0.9634 |
| Cohen's Kappa | 0.7841 |
| MCC | 0.7893 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 581.7 (1.72 ms/张) |
| End-to-End FPS | 233.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8731 | 0.9040 | 0.8190 | 0.8452 | 0.9347 | 0.9588 | 0.9370 | 197 |
| normal | 0.7899 | 0.9527 | 0.9356 | 0.8566 | 0.8389 | 0.9649 | 0.9552 | 276 |
| sever | 1.0000 | 0.9366 | 0.8000 | 0.8889 | 1.0000 | 0.9995 | 0.9979 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=172 FP=38 TN=358 FN=25
- **normal**: TP=218 FP=15 TN=302 FN=58
- **sever**: TP=120 FP=30 TN=443 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 172 | 15 | 10 |
| normal | 38 | 218 | 20 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8190    0.8731    0.8452       197
      normal     0.9356    0.7899    0.8566       276
       sever     0.8000    1.0000    0.8889       120

    accuracy                         0.8600       593
   macro avg     0.8516    0.8877    0.8636       593
weighted avg     0.8695    0.8600    0.8593       593

```