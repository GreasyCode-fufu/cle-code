# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r3/convnext_tiny_seed44/convnext_tiny/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8672 |
| Accuracy | 0.8617 |
| Macro-AUROC (OvR) | 0.9782 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8609 |
| Balanced Acc | 0.8907 |
| Macro-Sensitivity | 0.8907 |
| Macro-Specificity | 0.9323 |
| Macro-Precision | 0.8579 |
| AUC-PR (Macro) | 0.9677 |
| Cohen's Kappa | 0.7870 |
| MCC | 0.7945 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1701.1 (0.59 ms/张) |
| End-to-End FPS | 326.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9086 | 0.8813 | 0.7920 | 0.8463 | 0.9510 | 0.9674 | 0.9453 | 197 |
| normal | 0.7717 | 0.9685 | 0.9552 | 0.8537 | 0.8297 | 0.9696 | 0.9668 | 276 |
| sever | 0.9917 | 0.9471 | 0.8264 | 0.9015 | 0.9978 | 0.9977 | 0.9911 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=179 FP=47 TN=349 FN=18
- **normal**: TP=213 FP=10 TN=307 FN=63
- **sever**: TP=119 FP=25 TN=448 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 179 | 9 | 9 |
| normal | 47 | 213 | 16 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7920    0.9086    0.8463       197
      normal     0.9552    0.7717    0.8537       276
       sever     0.8264    0.9917    0.9015       120

    accuracy                         0.8617       593
   macro avg     0.8579    0.8907    0.8672       593
weighted avg     0.8749    0.8617    0.8609       593

```