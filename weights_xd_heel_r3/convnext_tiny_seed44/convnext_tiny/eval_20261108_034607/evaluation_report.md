# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r3/convnext_tiny_seed44/convnext_tiny/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8605 |
| Accuracy | 0.8567 |
| Macro-AUROC (OvR) | 0.9795 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8552 |
| Balanced Acc | 0.8901 |
| Macro-Sensitivity | 0.8901 |
| Macro-Specificity | 0.9314 |
| Macro-Precision | 0.8516 |
| AUC-PR (Macro) | 0.9705 |
| Cohen's Kappa | 0.7806 |
| MCC | 0.7910 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1704.1 (0.59 ms/张) |
| End-to-End FPS | 328.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9239 | 0.8838 | 0.7982 | 0.8565 | 0.9589 | 0.9669 | 0.9487 | 197 |
| normal | 0.7464 | 0.9779 | 0.9671 | 0.8425 | 0.8158 | 0.9719 | 0.9644 | 276 |
| sever | 1.0000 | 0.9323 | 0.7895 | 0.8824 | 1.0000 | 0.9996 | 0.9984 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=182 FP=46 TN=350 FN=15
- **normal**: TP=206 FP=7 TN=310 FN=70
- **sever**: TP=120 FP=32 TN=441 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 182 | 7 | 8 |
| normal | 46 | 206 | 24 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7982    0.9239    0.8565       197
      normal     0.9671    0.7464    0.8425       276
       sever     0.7895    1.0000    0.8824       120

    accuracy                         0.8567       593
   macro avg     0.8516    0.8901    0.8605       593
weighted avg     0.8751    0.8567    0.8552       593

```