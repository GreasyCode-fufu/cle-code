# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_svd_seed42/convnext_svd/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8808 |
| Accuracy | 0.8769 |
| Macro-AUROC (OvR) | 0.9705 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8762 |
| Balanced Acc | 0.8911 |
| Macro-Sensitivity | 0.8911 |
| Macro-Specificity | 0.9351 |
| Macro-Precision | 0.8733 |
| AUC-PR (Macro) | 0.9588 |
| Cohen's Kappa | 0.8069 |
| MCC | 0.8076 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 1485.4 (0.67 ms/张) |
| End-to-End FPS | 302.9 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8325 | 0.9369 | 0.8677 | 0.8497 | 0.9183 | 0.9530 | 0.9366 | 197 |
| normal | 0.8659 | 0.9085 | 0.8918 | 0.8787 | 0.8862 | 0.9609 | 0.9480 | 276 |
| sever | 0.9750 | 0.9598 | 0.8603 | 0.9141 | 0.9934 | 0.9977 | 0.9917 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=164 FP=25 TN=371 FN=33
- **normal**: TP=239 FP=29 TN=288 FN=37
- **sever**: TP=117 FP=19 TN=454 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 164 | 28 | 5 |
| normal | 23 | 239 | 14 |
| sever | 2 | 1 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8677    0.8325    0.8497       197
      normal     0.8918    0.8659    0.8787       276
       sever     0.8603    0.9750    0.9141       120

    accuracy                         0.8769       593
   macro avg     0.8733    0.8911    0.8808       593
weighted avg     0.8774    0.8769    0.8762       593

```