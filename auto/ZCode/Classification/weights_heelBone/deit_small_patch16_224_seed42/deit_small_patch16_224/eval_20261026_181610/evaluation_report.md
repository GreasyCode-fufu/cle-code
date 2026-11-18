# deit_small_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/deit_small_patch16_224_seed42/deit_small_patch16_224/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8276 |
| Accuracy | 0.8094 |
| Macro-AUROC (OvR) | 0.9498 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8076 |
| Balanced Acc | 0.8468 |
| Macro-Sensitivity | 0.8468 |
| Macro-Specificity | 0.9012 |
| Macro-Precision | 0.8213 |
| AUC-PR (Macro) | 0.9253 |
| Cohen's Kappa | 0.7057 |
| MCC | 0.7134 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.25G |
| FLOPs | 8.50G |
| Params | 21.59M |
| Pure-Inference FPS | 1014.2 (0.99 ms/张) |
| End-to-End FPS | 275.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8579 | 0.8258 | 0.7101 | 0.7770 | 0.9211 | 0.9301 | 0.8896 | 197 |
| normal | 0.6993 | 0.9117 | 0.8733 | 0.7767 | 0.7769 | 0.9219 | 0.8952 | 276 |
| sever | 0.9833 | 0.9662 | 0.8806 | 0.9291 | 0.9956 | 0.9975 | 0.9912 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=169 FP=69 TN=327 FN=28
- **normal**: TP=193 FP=28 TN=289 FN=83
- **sever**: TP=118 FP=16 TN=457 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 169 | 26 | 2 |
| normal | 69 | 193 | 14 |
| sever | 0 | 2 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7101    0.8579    0.7770       197
      normal     0.8733    0.6993    0.7767       276
       sever     0.8806    0.9833    0.9291       120

    accuracy                         0.8094       593
   macro avg     0.8213    0.8468    0.8276       593
weighted avg     0.8206    0.8094    0.8076       593

```