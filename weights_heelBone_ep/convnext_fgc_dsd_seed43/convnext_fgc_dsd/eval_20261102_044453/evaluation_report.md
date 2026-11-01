# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_fgc_dsd_seed43/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8641 |
| Accuracy | 0.8533 |
| Macro-AUROC (OvR) | 0.9730 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8517 |
| Balanced Acc | 0.8867 |
| Macro-Sensitivity | 0.8867 |
| Macro-Specificity | 0.9266 |
| Macro-Precision | 0.8566 |
| AUC-PR (Macro) | 0.9621 |
| Cohen's Kappa | 0.7740 |
| MCC | 0.7827 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 1087.1 (0.92 ms/张) |
| End-to-End FPS | 290.0 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9137 | 0.8611 | 0.7660 | 0.8333 | 0.9525 | 0.9587 | 0.9414 | 197 |
| normal | 0.7464 | 0.9590 | 0.9406 | 0.8323 | 0.8128 | 0.9615 | 0.9499 | 276 |
| sever | 1.0000 | 0.9598 | 0.8633 | 0.9266 | 1.0000 | 0.9988 | 0.9950 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=180 FP=55 TN=341 FN=17
- **normal**: TP=206 FP=13 TN=304 FN=70
- **sever**: TP=120 FP=19 TN=454 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 180 | 13 | 4 |
| normal | 55 | 206 | 15 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7660    0.9137    0.8333       197
      normal     0.9406    0.7464    0.8323       276
       sever     0.8633    1.0000    0.9266       120

    accuracy                         0.8533       593
   macro avg     0.8566    0.8867    0.8641       593
weighted avg     0.8670    0.8533    0.8517       593

```