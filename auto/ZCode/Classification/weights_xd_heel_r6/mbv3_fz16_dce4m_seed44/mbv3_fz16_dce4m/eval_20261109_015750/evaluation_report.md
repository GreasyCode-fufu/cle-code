# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce4m_seed44/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8454 |
| Accuracy | 0.8297 |
| Macro-AUROC (OvR) | 0.9426 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8292 |
| Balanced Acc | 0.8509 |
| Macro-Sensitivity | 0.8509 |
| Macro-Specificity | 0.9071 |
| Macro-Precision | 0.8405 |
| AUC-PR (Macro) | 0.9113 |
| Cohen's Kappa | 0.7320 |
| MCC | 0.7323 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3659.7 (0.27 ms/张) |
| End-to-End FPS | 18.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7817 | 0.8813 | 0.7662 | 0.7739 | 0.8903 | 0.9172 | 0.8506 | 197 |
| normal | 0.8043 | 0.8612 | 0.8346 | 0.8192 | 0.8349 | 0.9142 | 0.9014 | 276 |
| sever | 0.9667 | 0.9789 | 0.9206 | 0.9431 | 0.9914 | 0.9963 | 0.9819 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=154 FP=47 TN=349 FN=43
- **normal**: TP=222 FP=44 TN=273 FN=54
- **sever**: TP=116 FP=10 TN=463 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 154 | 42 | 1 |
| normal | 45 | 222 | 9 |
| sever | 2 | 2 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7662    0.7817    0.7739       197
      normal     0.8346    0.8043    0.8192       276
       sever     0.9206    0.9667    0.9431       120

    accuracy                         0.8297       593
   macro avg     0.8405    0.8509    0.8454       593
weighted avg     0.8293    0.8297    0.8292       593

```