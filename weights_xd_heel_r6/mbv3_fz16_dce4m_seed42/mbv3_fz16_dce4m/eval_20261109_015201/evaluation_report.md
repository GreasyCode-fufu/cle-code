# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce4m_seed42/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8528 |
| Accuracy | 0.8364 |
| Macro-AUROC (OvR) | 0.9457 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8367 |
| Balanced Acc | 0.8541 |
| Macro-Sensitivity | 0.8541 |
| Macro-Specificity | 0.9102 |
| Macro-Precision | 0.8518 |
| AUC-PR (Macro) | 0.9184 |
| Cohen's Kappa | 0.7419 |
| MCC | 0.7422 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3511.0 (0.28 ms/张) |
| End-to-End FPS | 18.6 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7970 | 0.8788 | 0.7659 | 0.7811 | 0.8969 | 0.9214 | 0.8619 | 197 |
| normal | 0.8152 | 0.8644 | 0.8396 | 0.8272 | 0.8431 | 0.9190 | 0.9062 | 276 |
| sever | 0.9500 | 0.9873 | 0.9500 | 0.9500 | 0.9873 | 0.9966 | 0.9871 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=157 FP=48 TN=348 FN=40
- **normal**: TP=225 FP=43 TN=274 FN=51
- **sever**: TP=114 FP=6 TN=467 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 157 | 40 | 0 |
| normal | 45 | 225 | 6 |
| sever | 3 | 3 | 114 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7659    0.7970    0.7811       197
      normal     0.8396    0.8152    0.8272       276
       sever     0.9500    0.9500    0.9500       120

    accuracy                         0.8364       593
   macro avg     0.8518    0.8541    0.8528       593
weighted avg     0.8374    0.8364    0.8367       593

```