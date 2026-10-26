# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/convnext_v2c_seed42/convnext_v2c/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8736 |
| Accuracy | 0.8567 |
| Macro-AUROC (OvR) | 0.9697 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8571 |
| Balanced Acc | 0.8828 |
| Macro-Sensitivity | 0.8828 |
| Macro-Specificity | 0.9252 |
| Macro-Precision | 0.8744 |
| AUC-PR (Macro) | 0.9570 |
| Cohen's Kappa | 0.7769 |
| MCC | 0.7838 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 741.6 (1.35 ms/张) |
| End-to-End FPS | 255.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9137 | 0.8460 | 0.7469 | 0.8219 | 0.9517 | 0.9564 | 0.9272 | 197 |
| normal | 0.7681 | 0.9401 | 0.9177 | 0.8363 | 0.8232 | 0.9539 | 0.9486 | 276 |
| sever | 0.9667 | 0.9894 | 0.9587 | 0.9627 | 0.9915 | 0.9988 | 0.9952 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=180 FP=61 TN=335 FN=17
- **normal**: TP=212 FP=19 TN=298 FN=64
- **sever**: TP=116 FP=5 TN=468 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 180 | 16 | 1 |
| normal | 60 | 212 | 4 |
| sever | 1 | 3 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7469    0.9137    0.8219       197
      normal     0.9177    0.7681    0.8363       276
       sever     0.9587    0.9667    0.9627       120

    accuracy                         0.8567       593
   macro avg     0.8744    0.8828    0.8736       593
weighted avg     0.8693    0.8567    0.8571       593

```