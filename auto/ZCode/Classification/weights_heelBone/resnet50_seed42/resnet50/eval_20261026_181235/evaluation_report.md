# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/resnet50_seed42/resnet50/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8015 |
| Accuracy | 0.8061 |
| Macro-AUROC (OvR) | 0.9640 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8107 |
| Balanced Acc | 0.8363 |
| Macro-Sensitivity | 0.8363 |
| Macro-Specificity | 0.9087 |
| Macro-Precision | 0.8062 |
| AUC-PR (Macro) | 0.9478 |
| Cohen's Kappa | 0.7053 |
| MCC | 0.7190 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 1120.4 (0.89 ms/张) |
| End-to-End FPS | 286.0 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7563 | 0.9646 | 0.9141 | 0.8278 | 0.8884 | 0.9553 | 0.9282 | 197 |
| normal | 0.7609 | 0.9243 | 0.8974 | 0.8235 | 0.8162 | 0.9443 | 0.9398 | 276 |
| sever | 0.9917 | 0.8372 | 0.6071 | 0.7532 | 0.9975 | 0.9925 | 0.9755 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=149 FP=14 TN=382 FN=48
- **normal**: TP=210 FP=24 TN=293 FN=66
- **sever**: TP=119 FP=77 TN=396 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 149 | 23 | 25 |
| normal | 14 | 210 | 52 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.9141    0.7563    0.8278       197
      normal     0.8974    0.7609    0.8235       276
       sever     0.6071    0.9917    0.7532       120

    accuracy                         0.8061       593
   macro avg     0.8062    0.8363    0.8015       593
weighted avg     0.8442    0.8061    0.8107       593

```