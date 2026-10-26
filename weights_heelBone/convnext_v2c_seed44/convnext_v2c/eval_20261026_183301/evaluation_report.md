# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/convnext_v2c_seed44/convnext_v2c/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9051 |
| Accuracy | 0.8971 |
| Macro-AUROC (OvR) | 0.9793 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8968 |
| Balanced Acc | 0.9081 |
| Macro-Sensitivity | 0.9081 |
| Macro-Specificity | 0.9435 |
| Macro-Precision | 0.9025 |
| AUC-PR (Macro) | 0.9699 |
| Cohen's Kappa | 0.8375 |
| MCC | 0.8376 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1652.3 (0.61 ms/张) |
| End-to-End FPS | 331.6 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8579 | 0.9419 | 0.8802 | 0.8689 | 0.9302 | 0.9682 | 0.9507 | 197 |
| normal | 0.8913 | 0.9054 | 0.8913 | 0.8913 | 0.9054 | 0.9714 | 0.9653 | 276 |
| sever | 0.9750 | 0.9831 | 0.9360 | 0.9551 | 0.9936 | 0.9983 | 0.9939 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=169 FP=23 TN=373 FN=28
- **normal**: TP=246 FP=30 TN=287 FN=30
- **sever**: TP=117 FP=8 TN=465 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 169 | 27 | 1 |
| normal | 23 | 246 | 7 |
| sever | 0 | 3 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8802    0.8579    0.8689       197
      normal     0.8913    0.8913    0.8913       276
       sever     0.9360    0.9750    0.9551       120

    accuracy                         0.8971       593
   macro avg     0.9025    0.9081    0.9051       593
weighted avg     0.8967    0.8971    0.8968       593

```