# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_dce4m_seed42/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8591 |
| Accuracy | 0.8465 |
| Macro-AUROC (OvR) | 0.9534 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8460 |
| Balanced Acc | 0.8624 |
| Macro-Sensitivity | 0.8624 |
| Macro-Specificity | 0.9155 |
| Macro-Precision | 0.8561 |
| AUC-PR (Macro) | 0.9296 |
| Cohen's Kappa | 0.7577 |
| MCC | 0.7578 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3341.5 (0.30 ms/张) |
| End-to-End FPS | 18.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7919 | 0.9066 | 0.8083 | 0.8000 | 0.8975 | 0.9340 | 0.8805 | 197 |
| normal | 0.8370 | 0.8612 | 0.8400 | 0.8385 | 0.8585 | 0.9284 | 0.9172 | 276 |
| sever | 0.9583 | 0.9789 | 0.9200 | 0.9388 | 0.9893 | 0.9977 | 0.9912 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=156 FP=37 TN=359 FN=41
- **normal**: TP=231 FP=44 TN=273 FN=45
- **sever**: TP=115 FP=10 TN=463 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 156 | 40 | 1 |
| normal | 36 | 231 | 9 |
| sever | 1 | 4 | 115 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8083    0.7919    0.8000       197
      normal     0.8400    0.8370    0.8385       276
       sever     0.9200    0.9583    0.9388       120

    accuracy                         0.8465       593
   macro avg     0.8561    0.8624    0.8591       593
weighted avg     0.8457    0.8465    0.8460       593

```