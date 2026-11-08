# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz8m_seed42/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7592 |
| Accuracy | 0.7487 |
| Macro-AUROC (OvR) | 0.9355 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7424 |
| Balanced Acc | 0.8009 |
| Macro-Sensitivity | 0.8009 |
| Macro-Specificity | 0.8761 |
| Macro-Precision | 0.7558 |
| AUC-PR (Macro) | 0.9043 |
| Cohen's Kappa | 0.6205 |
| MCC | 0.6377 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2748.3 (0.36 ms/张) |
| End-to-End FPS | 31.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8325 | 0.7980 | 0.6721 | 0.7438 | 0.9054 | 0.9090 | 0.8464 | 197 |
| normal | 0.5870 | 0.9274 | 0.8757 | 0.7028 | 0.7206 | 0.9051 | 0.8934 | 276 |
| sever | 0.9833 | 0.9027 | 0.7195 | 0.8310 | 0.9953 | 0.9922 | 0.9733 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=164 FP=80 TN=316 FN=33
- **normal**: TP=162 FP=23 TN=294 FN=114
- **sever**: TP=118 FP=46 TN=427 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 164 | 21 | 12 |
| normal | 80 | 162 | 34 |
| sever | 0 | 2 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6721    0.8325    0.7438       197
      normal     0.8757    0.5870    0.7028       276
       sever     0.7195    0.9833    0.8310       120

    accuracy                         0.7487       593
   macro avg     0.7558    0.8009    0.7592       593
weighted avg     0.7765    0.7487    0.7424       593

```