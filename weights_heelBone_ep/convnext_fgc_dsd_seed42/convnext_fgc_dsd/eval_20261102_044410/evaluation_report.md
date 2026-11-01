# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_fgc_dsd_seed42/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8778 |
| Accuracy | 0.8752 |
| Macro-AUROC (OvR) | 0.9742 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8746 |
| Balanced Acc | 0.8961 |
| Macro-Sensitivity | 0.8961 |
| Macro-Specificity | 0.9366 |
| Macro-Precision | 0.8668 |
| AUC-PR (Macro) | 0.9628 |
| Cohen's Kappa | 0.8058 |
| MCC | 0.8081 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 941.9 (1.06 ms/张) |
| End-to-End FPS | 278.2 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8477 | 0.9394 | 0.8743 | 0.8608 | 0.9254 | 0.9623 | 0.9410 | 197 |
| normal | 0.8406 | 0.9274 | 0.9098 | 0.8738 | 0.8698 | 0.9623 | 0.9561 | 276 |
| sever | 1.0000 | 0.9429 | 0.8163 | 0.8989 | 1.0000 | 0.9981 | 0.9912 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=167 FP=24 TN=372 FN=30
- **normal**: TP=232 FP=23 TN=294 FN=44
- **sever**: TP=120 FP=27 TN=446 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 167 | 23 | 7 |
| normal | 24 | 232 | 20 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8743    0.8477    0.8608       197
      normal     0.9098    0.8406    0.8738       276
       sever     0.8163    1.0000    0.8989       120

    accuracy                         0.8752       593
   macro avg     0.8668    0.8961    0.8778       593
weighted avg     0.8791    0.8752    0.8746       593

```