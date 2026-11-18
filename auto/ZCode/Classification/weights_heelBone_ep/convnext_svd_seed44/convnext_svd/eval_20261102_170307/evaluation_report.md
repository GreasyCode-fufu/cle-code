# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_svd_seed44/convnext_svd/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8685 |
| Accuracy | 0.8583 |
| Macro-AUROC (OvR) | 0.9719 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8573 |
| Balanced Acc | 0.8864 |
| Macro-Sensitivity | 0.8864 |
| Macro-Specificity | 0.9274 |
| Macro-Precision | 0.8580 |
| AUC-PR (Macro) | 0.9594 |
| Cohen's Kappa | 0.7802 |
| MCC | 0.7841 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 1018.4 (0.98 ms/张) |
| End-to-End FPS | 235.2 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8731 | 0.8864 | 0.7926 | 0.8309 | 0.9335 | 0.9575 | 0.9369 | 197 |
| normal | 0.7862 | 0.9338 | 0.9118 | 0.8444 | 0.8338 | 0.9602 | 0.9498 | 276 |
| sever | 1.0000 | 0.9619 | 0.8696 | 0.9302 | 1.0000 | 0.9981 | 0.9917 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=172 FP=45 TN=351 FN=25
- **normal**: TP=217 FP=21 TN=296 FN=59
- **sever**: TP=120 FP=18 TN=455 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 172 | 21 | 4 |
| normal | 45 | 217 | 14 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7926    0.8731    0.8309       197
      normal     0.9118    0.7862    0.8444       276
       sever     0.8696    1.0000    0.9302       120

    accuracy                         0.8583       593
   macro avg     0.8580    0.8864    0.8685       593
weighted avg     0.8636    0.8583    0.8573       593

```