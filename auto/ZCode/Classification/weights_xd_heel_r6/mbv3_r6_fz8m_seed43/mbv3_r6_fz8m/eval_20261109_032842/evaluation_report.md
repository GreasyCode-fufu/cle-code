# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz8m_seed43/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7909 |
| Accuracy | 0.7858 |
| Macro-AUROC (OvR) | 0.9357 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7846 |
| Balanced Acc | 0.8185 |
| Macro-Sensitivity | 0.8185 |
| Macro-Specificity | 0.8926 |
| Macro-Precision | 0.7790 |
| AUC-PR (Macro) | 0.9030 |
| Cohen's Kappa | 0.6711 |
| MCC | 0.6769 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4026.5 (0.25 ms/张) |
| End-to-End FPS | 170.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7716 | 0.8611 | 0.7343 | 0.7525 | 0.8834 | 0.9087 | 0.8412 | 197 |
| normal | 0.7174 | 0.9054 | 0.8684 | 0.7857 | 0.7863 | 0.9066 | 0.8921 | 276 |
| sever | 0.9667 | 0.9112 | 0.7342 | 0.8345 | 0.9908 | 0.9917 | 0.9756 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=152 FP=55 TN=341 FN=45
- **normal**: TP=198 FP=30 TN=287 FN=78
- **sever**: TP=116 FP=42 TN=431 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 152 | 27 | 18 |
| normal | 54 | 198 | 24 |
| sever | 1 | 3 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7343    0.7716    0.7525       197
      normal     0.8684    0.7174    0.7857       276
       sever     0.7342    0.9667    0.8345       120

    accuracy                         0.7858       593
   macro avg     0.7790    0.8185    0.7909       593
weighted avg     0.7967    0.7858    0.7846       593

```