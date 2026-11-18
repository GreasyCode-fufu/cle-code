# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_abl/convnext_v2m_seed42/convnext_v2m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8584 |
| Accuracy | 0.8583 |
| Macro-AUROC (OvR) | 0.9797 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8583 |
| Balanced Acc | 0.8849 |
| Macro-Sensitivity | 0.8849 |
| Macro-Specificity | 0.9313 |
| Macro-Precision | 0.8468 |
| AUC-PR (Macro) | 0.9704 |
| Cohen's Kappa | 0.7820 |
| MCC | 0.7877 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1651.9 (0.61 ms/张) |
| End-to-End FPS | 323.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8731 | 0.9268 | 0.8557 | 0.8643 | 0.9362 | 0.9712 | 0.9537 | 197 |
| normal | 0.7899 | 0.9495 | 0.9316 | 0.8549 | 0.8384 | 0.9701 | 0.9639 | 276 |
| sever | 0.9917 | 0.9175 | 0.7532 | 0.8561 | 0.9977 | 0.9979 | 0.9937 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=172 FP=29 TN=367 FN=25
- **normal**: TP=218 FP=16 TN=301 FN=58
- **sever**: TP=119 FP=39 TN=434 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 172 | 15 | 10 |
| normal | 29 | 218 | 29 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8557    0.8731    0.8643       197
      normal     0.9316    0.7899    0.8549       276
       sever     0.7532    0.9917    0.8561       120

    accuracy                         0.8583       593
   macro avg     0.8468    0.8849    0.8584       593
weighted avg     0.8703    0.8583    0.8583       593

```