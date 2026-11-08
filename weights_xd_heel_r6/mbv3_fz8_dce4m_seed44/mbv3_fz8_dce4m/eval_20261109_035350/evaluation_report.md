# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz8_dce4m_seed44/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8681 |
| Accuracy | 0.8583 |
| Macro-AUROC (OvR) | 0.9551 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8569 |
| Balanced Acc | 0.8731 |
| Macro-Sensitivity | 0.8731 |
| Macro-Specificity | 0.9222 |
| Macro-Precision | 0.8647 |
| AUC-PR (Macro) | 0.9353 |
| Cohen's Kappa | 0.7762 |
| MCC | 0.7767 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4222.8 (0.24 ms/张) |
| End-to-End FPS | 13.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7665 | 0.9192 | 0.8251 | 0.7947 | 0.8878 | 0.9294 | 0.8938 | 197 |
| normal | 0.8696 | 0.8707 | 0.8541 | 0.8618 | 0.8846 | 0.9379 | 0.9201 | 276 |
| sever | 0.9833 | 0.9767 | 0.9147 | 0.9478 | 0.9957 | 0.9979 | 0.9920 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=151 FP=32 TN=364 FN=46
- **normal**: TP=240 FP=41 TN=276 FN=36
- **sever**: TP=118 FP=11 TN=462 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 151 | 41 | 5 |
| normal | 30 | 240 | 6 |
| sever | 2 | 0 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8251    0.7665    0.7947       197
      normal     0.8541    0.8696    0.8618       276
       sever     0.9147    0.9833    0.9478       120

    accuracy                         0.8583       593
   macro avg     0.8647    0.8731    0.8681       593
weighted avg     0.8567    0.8583    0.8569       593

```