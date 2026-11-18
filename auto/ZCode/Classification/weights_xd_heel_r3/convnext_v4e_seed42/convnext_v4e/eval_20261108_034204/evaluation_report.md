# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r3/convnext_v4e_seed42/convnext_v4e/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8992 |
| Accuracy | 0.8954 |
| Macro-AUROC (OvR) | 0.9802 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8948 |
| Balanced Acc | 0.9090 |
| Macro-Sensitivity | 0.9090 |
| Macro-Specificity | 0.9450 |
| Macro-Precision | 0.8920 |
| AUC-PR (Macro) | 0.9701 |
| Cohen's Kappa | 0.8358 |
| MCC | 0.8365 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 433.4 (2.31 ms/张) |
| End-to-End FPS | 204.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8477 | 0.9444 | 0.8836 | 0.8653 | 0.9257 | 0.9704 | 0.9536 | 197 |
| normal | 0.8877 | 0.9243 | 0.9108 | 0.8991 | 0.9043 | 0.9714 | 0.9616 | 276 |
| sever | 0.9917 | 0.9662 | 0.8815 | 0.9333 | 0.9978 | 0.9987 | 0.9951 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=167 FP=22 TN=374 FN=30
- **normal**: TP=245 FP=24 TN=293 FN=31
- **sever**: TP=119 FP=16 TN=457 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 167 | 23 | 7 |
| normal | 22 | 245 | 9 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8836    0.8477    0.8653       197
      normal     0.9108    0.8877    0.8991       276
       sever     0.8815    0.9917    0.9333       120

    accuracy                         0.8954       593
   macro avg     0.8920    0.9090    0.8992       593
weighted avg     0.8958    0.8954    0.8948       593

```