# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mobilenet_v3_large_seed42/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8264 |
| Accuracy | 0.8078 |
| Macro-AUROC (OvR) | 0.9490 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8049 |
| Balanced Acc | 0.8494 |
| Macro-Sensitivity | 0.8494 |
| Macro-Specificity | 0.9023 |
| Macro-Precision | 0.8287 |
| AUC-PR (Macro) | 0.9215 |
| Cohen's Kappa | 0.7050 |
| MCC | 0.7208 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 2691.3 (0.37 ms/张) |
| End-to-End FPS | 153.9 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9137 | 0.7955 | 0.6897 | 0.7860 | 0.9488 | 0.9306 | 0.8744 | 197 |
| normal | 0.6594 | 0.9432 | 0.9100 | 0.7647 | 0.7608 | 0.9203 | 0.9039 | 276 |
| sever | 0.9750 | 0.9683 | 0.8864 | 0.9286 | 0.9935 | 0.9962 | 0.9862 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=180 FP=81 TN=315 FN=17
- **normal**: TP=182 FP=18 TN=299 FN=94
- **sever**: TP=117 FP=15 TN=458 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 180 | 15 | 2 |
| normal | 81 | 182 | 13 |
| sever | 0 | 3 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6897    0.9137    0.7860       197
      normal     0.9100    0.6594    0.7647       276
       sever     0.8864    0.9750    0.9286       120

    accuracy                         0.8078       593
   macro avg     0.8287    0.8494    0.8264       593
weighted avg     0.8320    0.8078    0.8049       593

```