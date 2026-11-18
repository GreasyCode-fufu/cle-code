# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz16_seed43/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7492 |
| Accuracy | 0.7268 |
| Macro-AUROC (OvR) | 0.9250 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7217 |
| Balanced Acc | 0.7791 |
| Macro-Sensitivity | 0.7791 |
| Macro-Specificity | 0.8607 |
| Macro-Precision | 0.7514 |
| AUC-PR (Macro) | 0.8848 |
| Cohen's Kappa | 0.5841 |
| MCC | 0.6015 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3007.8 (0.33 ms/张) |
| End-to-End FPS | 213.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8173 | 0.7348 | 0.6053 | 0.6955 | 0.8899 | 0.8869 | 0.8155 | 197 |
| normal | 0.5616 | 0.9022 | 0.8333 | 0.6710 | 0.7027 | 0.8960 | 0.8640 | 276 |
| sever | 0.9583 | 0.9450 | 0.8156 | 0.8812 | 0.9889 | 0.9921 | 0.9749 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=161 FP=105 TN=291 FN=36
- **normal**: TP=155 FP=31 TN=286 FN=121
- **sever**: TP=115 FP=26 TN=447 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 161 | 28 | 8 |
| normal | 103 | 155 | 18 |
| sever | 2 | 3 | 115 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6053    0.8173    0.6955       197
      normal     0.8333    0.5616    0.6710       276
       sever     0.8156    0.9583    0.8812       120

    accuracy                         0.7268       593
   macro avg     0.7514    0.7791    0.7492       593
weighted avg     0.7540    0.7268    0.7217       593

```