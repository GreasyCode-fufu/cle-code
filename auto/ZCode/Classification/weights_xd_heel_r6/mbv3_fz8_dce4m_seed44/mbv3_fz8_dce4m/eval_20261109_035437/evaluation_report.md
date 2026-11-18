# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz8_dce4m_seed44/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8614 |
| Accuracy | 0.8449 |
| Macro-AUROC (OvR) | 0.9524 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8447 |
| Balanced Acc | 0.8663 |
| Macro-Sensitivity | 0.8663 |
| Macro-Specificity | 0.9153 |
| Macro-Precision | 0.8573 |
| AUC-PR (Macro) | 0.9315 |
| Cohen's Kappa | 0.7560 |
| MCC | 0.7565 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4111.1 (0.24 ms/张) |
| End-to-End FPS | 13.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8122 | 0.8838 | 0.7767 | 0.7940 | 0.9044 | 0.9293 | 0.8871 | 197 |
| normal | 0.8116 | 0.8770 | 0.8517 | 0.8312 | 0.8424 | 0.9293 | 0.9127 | 276 |
| sever | 0.9750 | 0.9852 | 0.9435 | 0.9590 | 0.9936 | 0.9985 | 0.9948 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=160 FP=46 TN=350 FN=37
- **normal**: TP=224 FP=39 TN=278 FN=52
- **sever**: TP=117 FP=7 TN=466 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 160 | 36 | 1 |
| normal | 46 | 224 | 6 |
| sever | 0 | 3 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7767    0.8122    0.7940       197
      normal     0.8517    0.8116    0.8312       276
       sever     0.9435    0.9750    0.9590       120

    accuracy                         0.8449       593
   macro avg     0.8573    0.8663    0.8614       593
weighted avg     0.8454    0.8449    0.8447       593

```