# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce4s_seed44/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8358 |
| Accuracy | 0.8212 |
| Macro-AUROC (OvR) | 0.9349 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8203 |
| Balanced Acc | 0.8374 |
| Macro-Sensitivity | 0.8374 |
| Macro-Specificity | 0.9006 |
| Macro-Precision | 0.8349 |
| AUC-PR (Macro) | 0.9022 |
| Cohen's Kappa | 0.7169 |
| MCC | 0.7172 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3898.7 (0.26 ms/张) |
| End-to-End FPS | 18.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7360 | 0.8965 | 0.7796 | 0.7572 | 0.8722 | 0.9050 | 0.8450 | 197 |
| normal | 0.8261 | 0.8265 | 0.8057 | 0.8157 | 0.8452 | 0.9038 | 0.8760 | 276 |
| sever | 0.9500 | 0.9789 | 0.9194 | 0.9344 | 0.9872 | 0.9959 | 0.9855 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=145 FP=41 TN=355 FN=52
- **normal**: TP=228 FP=55 TN=262 FN=48
- **sever**: TP=114 FP=10 TN=463 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 145 | 50 | 2 |
| normal | 40 | 228 | 8 |
| sever | 1 | 5 | 114 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7796    0.7360    0.7572       197
      normal     0.8057    0.8261    0.8157       276
       sever     0.9194    0.9500    0.9344       120

    accuracy                         0.8212       593
   macro avg     0.8349    0.8374    0.8358       593
weighted avg     0.8200    0.8212    0.8203       593

```