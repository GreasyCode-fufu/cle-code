# deit_small_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/deit_small_patch16_224_seed44/deit_small_patch16_224/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7724 |
| Accuracy | 0.7487 |
| Macro-AUROC (OvR) | 0.9404 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7440 |
| Balanced Acc | 0.7977 |
| Macro-Sensitivity | 0.7977 |
| Macro-Specificity | 0.8731 |
| Macro-Precision | 0.7957 |
| AUC-PR (Macro) | 0.9103 |
| Cohen's Kappa | 0.6177 |
| MCC | 0.6475 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.25G |
| FLOPs | 8.50G |
| Params | 21.59M |
| Pure-Inference FPS | 933.3 (1.07 ms/张) |
| End-to-End FPS | 275.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9137 | 0.7045 | 0.6061 | 0.7287 | 0.9426 | 0.9140 | 0.8637 | 197 |
| normal | 0.5543 | 0.9464 | 0.9000 | 0.6861 | 0.7092 | 0.9146 | 0.8901 | 276 |
| sever | 0.9250 | 0.9683 | 0.8810 | 0.9024 | 0.9807 | 0.9926 | 0.9772 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=180 FP=117 TN=279 FN=17
- **normal**: TP=153 FP=17 TN=300 FN=123
- **sever**: TP=111 FP=15 TN=458 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 180 | 12 | 5 |
| normal | 113 | 153 | 10 |
| sever | 4 | 5 | 111 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6061    0.9137    0.7287       197
      normal     0.9000    0.5543    0.6861       276
       sever     0.8810    0.9250    0.9024       120

    accuracy                         0.7487       593
   macro avg     0.7957    0.7977    0.7724       593
weighted avg     0.7985    0.7487    0.7440       593

```