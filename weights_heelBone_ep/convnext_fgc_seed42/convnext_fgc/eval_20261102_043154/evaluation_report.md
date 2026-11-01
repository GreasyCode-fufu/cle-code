# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_fgc_seed42/convnext_fgc/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8363 |
| Accuracy | 0.8364 |
| Macro-AUROC (OvR) | 0.9749 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8365 |
| Balanced Acc | 0.8693 |
| Macro-Sensitivity | 0.8693 |
| Macro-Specificity | 0.9216 |
| Macro-Precision | 0.8265 |
| AUC-PR (Macro) | 0.9635 |
| Cohen's Kappa | 0.7501 |
| MCC | 0.7589 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 1010.8 (0.99 ms/张) |
| End-to-End FPS | 266.0 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8579 | 0.9293 | 0.8579 | 0.8579 | 0.9293 | 0.9663 | 0.9494 | 197 |
| normal | 0.7500 | 0.9432 | 0.9200 | 0.8263 | 0.8125 | 0.9608 | 0.9500 | 276 |
| sever | 1.0000 | 0.8922 | 0.7018 | 0.8247 | 1.0000 | 0.9977 | 0.9910 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=169 FP=28 TN=368 FN=28
- **normal**: TP=207 FP=18 TN=299 FN=69
- **sever**: TP=120 FP=51 TN=422 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 169 | 18 | 10 |
| normal | 28 | 207 | 41 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8579    0.8579    0.8579       197
      normal     0.9200    0.7500    0.8263       276
       sever     0.7018    1.0000    0.8247       120

    accuracy                         0.8364       593
   macro avg     0.8265    0.8693    0.8363       593
weighted avg     0.8552    0.8364    0.8365       593

```