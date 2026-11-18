# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/resnet50_seed44/resnet50/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8382 |
| Accuracy | 0.8347 |
| Macro-AUROC (OvR) | 0.9639 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8340 |
| Balanced Acc | 0.8603 |
| Macro-Sensitivity | 0.8603 |
| Macro-Specificity | 0.9170 |
| Macro-Precision | 0.8264 |
| AUC-PR (Macro) | 0.9475 |
| Cohen's Kappa | 0.7446 |
| MCC | 0.7488 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 1138.6 (0.88 ms/张) |
| End-to-End FPS | 285.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8426 | 0.8965 | 0.8019 | 0.8218 | 0.9197 | 0.9536 | 0.9304 | 197 |
| normal | 0.7717 | 0.9243 | 0.8987 | 0.8304 | 0.8230 | 0.9471 | 0.9404 | 276 |
| sever | 0.9667 | 0.9302 | 0.7785 | 0.8625 | 0.9910 | 0.9909 | 0.9716 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=166 FP=41 TN=355 FN=31
- **normal**: TP=213 FP=24 TN=293 FN=63
- **sever**: TP=116 FP=33 TN=440 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 166 | 20 | 11 |
| normal | 41 | 213 | 22 |
| sever | 0 | 4 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8019    0.8426    0.8218       197
      normal     0.8987    0.7717    0.8304       276
       sever     0.7785    0.9667    0.8625       120

    accuracy                         0.8347       593
   macro avg     0.8264    0.8603    0.8382       593
weighted avg     0.8422    0.8347    0.8340       593

```