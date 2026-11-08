# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz8_dce4m_seed42/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8625 |
| Accuracy | 0.8482 |
| Macro-AUROC (OvR) | 0.9457 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8475 |
| Balanced Acc | 0.8658 |
| Macro-Sensitivity | 0.8658 |
| Macro-Specificity | 0.9161 |
| Macro-Precision | 0.8596 |
| AUC-PR (Macro) | 0.9180 |
| Cohen's Kappa | 0.7603 |
| MCC | 0.7604 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3375.9 (0.30 ms/张) |
| End-to-End FPS | 8.2 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7817 | 0.9040 | 0.8021 | 0.7918 | 0.8928 | 0.9207 | 0.8721 | 197 |
| normal | 0.8406 | 0.8612 | 0.8406 | 0.8406 | 0.8612 | 0.9204 | 0.9012 | 276 |
| sever | 0.9750 | 0.9831 | 0.9360 | 0.9551 | 0.9936 | 0.9961 | 0.9808 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=154 FP=38 TN=358 FN=43
- **normal**: TP=232 FP=44 TN=273 FN=44
- **sever**: TP=117 FP=8 TN=465 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 154 | 43 | 0 |
| normal | 36 | 232 | 8 |
| sever | 2 | 1 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8021    0.7817    0.7918       197
      normal     0.8406    0.8406    0.8406       276
       sever     0.9360    0.9750    0.9551       120

    accuracy                         0.8482       593
   macro avg     0.8596    0.8658    0.8625       593
weighted avg     0.8471    0.8482    0.8475       593

```