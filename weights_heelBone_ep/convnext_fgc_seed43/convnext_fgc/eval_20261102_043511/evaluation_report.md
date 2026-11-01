# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_fgc_seed43/convnext_fgc/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8816 |
| Accuracy | 0.8769 |
| Macro-AUROC (OvR) | 0.9740 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8753 |
| Balanced Acc | 0.8915 |
| Macro-Sensitivity | 0.8915 |
| Macro-Specificity | 0.9334 |
| Macro-Precision | 0.8778 |
| AUC-PR (Macro) | 0.9633 |
| Cohen's Kappa | 0.8061 |
| MCC | 0.8080 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 1011.8 (0.99 ms/张) |
| End-to-End FPS | 284.7 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7868 | 0.9571 | 0.9012 | 0.8401 | 0.9002 | 0.9581 | 0.9397 | 197 |
| normal | 0.8877 | 0.8833 | 0.8688 | 0.8781 | 0.9003 | 0.9648 | 0.9531 | 276 |
| sever | 1.0000 | 0.9598 | 0.8633 | 0.9266 | 1.0000 | 0.9992 | 0.9971 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=155 FP=17 TN=379 FN=42
- **normal**: TP=245 FP=37 TN=280 FN=31
- **sever**: TP=120 FP=19 TN=454 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 155 | 37 | 5 |
| normal | 17 | 245 | 14 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.9012    0.7868    0.8401       197
      normal     0.8688    0.8877    0.8781       276
       sever     0.8633    1.0000    0.9266       120

    accuracy                         0.8769       593
   macro avg     0.8778    0.8915    0.8816       593
weighted avg     0.8784    0.8769    0.8753       593

```