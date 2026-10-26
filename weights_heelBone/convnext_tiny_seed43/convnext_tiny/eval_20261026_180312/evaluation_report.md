# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/convnext_tiny_seed43/convnext_tiny/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8821 |
| Accuracy | 0.8718 |
| Macro-AUROC (OvR) | 0.9730 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8720 |
| Balanced Acc | 0.8926 |
| Macro-Sensitivity | 0.8926 |
| Macro-Specificity | 0.9340 |
| Macro-Precision | 0.8787 |
| AUC-PR (Macro) | 0.9596 |
| Cohen's Kappa | 0.8003 |
| MCC | 0.8052 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 772.0 (1.30 ms/张) |
| End-to-End FPS | 262.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9188 | 0.8737 | 0.7835 | 0.8458 | 0.9558 | 0.9614 | 0.9395 | 197 |
| normal | 0.8007 | 0.9495 | 0.9325 | 0.8616 | 0.8455 | 0.9630 | 0.9569 | 276 |
| sever | 0.9583 | 0.9789 | 0.9200 | 0.9388 | 0.9893 | 0.9946 | 0.9823 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=181 FP=50 TN=346 FN=16
- **normal**: TP=221 FP=16 TN=301 FN=55
- **sever**: TP=115 FP=10 TN=463 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 181 | 13 | 3 |
| normal | 48 | 221 | 7 |
| sever | 2 | 3 | 115 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7835    0.9188    0.8458       197
      normal     0.9325    0.8007    0.8616       276
       sever     0.9200    0.9583    0.9388       120

    accuracy                         0.8718       593
   macro avg     0.8787    0.8926    0.8821       593
weighted avg     0.8805    0.8718    0.8720       593

```