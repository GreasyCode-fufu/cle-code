# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz8m_seed42/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7862 |
| Accuracy | 0.7707 |
| Macro-AUROC (OvR) | 0.9326 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7679 |
| Balanced Acc | 0.8119 |
| Macro-Sensitivity | 0.8119 |
| Macro-Specificity | 0.8833 |
| Macro-Precision | 0.7811 |
| AUC-PR (Macro) | 0.8960 |
| Cohen's Kappa | 0.6486 |
| MCC | 0.6597 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2454.8 (0.41 ms/张) |
| End-to-End FPS | 27.6 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8325 | 0.7955 | 0.6694 | 0.7421 | 0.9052 | 0.9039 | 0.8409 | 197 |
| normal | 0.6449 | 0.9117 | 0.8641 | 0.7386 | 0.7468 | 0.9026 | 0.8824 | 276 |
| sever | 0.9583 | 0.9429 | 0.8099 | 0.8779 | 0.9889 | 0.9913 | 0.9647 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=164 FP=81 TN=315 FN=33
- **normal**: TP=178 FP=28 TN=289 FN=98
- **sever**: TP=115 FP=27 TN=446 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 164 | 25 | 8 |
| normal | 79 | 178 | 19 |
| sever | 2 | 3 | 115 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6694    0.8325    0.7421       197
      normal     0.8641    0.6449    0.7386       276
       sever     0.8099    0.9583    0.8779       120

    accuracy                         0.7707       593
   macro avg     0.7811    0.8119    0.7862       593
weighted avg     0.7884    0.7707    0.7679       593

```