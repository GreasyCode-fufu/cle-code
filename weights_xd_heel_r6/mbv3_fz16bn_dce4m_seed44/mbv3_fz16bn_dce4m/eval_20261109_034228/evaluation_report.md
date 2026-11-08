# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16bn_dce4m_seed44/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8341 |
| Accuracy | 0.8196 |
| Macro-AUROC (OvR) | 0.9298 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8180 |
| Balanced Acc | 0.8379 |
| Macro-Sensitivity | 0.8379 |
| Macro-Specificity | 0.8999 |
| Macro-Precision | 0.8317 |
| AUC-PR (Macro) | 0.8912 |
| Cohen's Kappa | 0.7145 |
| MCC | 0.7150 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4139.8 (0.24 ms/张) |
| End-to-End FPS | 13.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7208 | 0.8965 | 0.7760 | 0.7474 | 0.8659 | 0.8967 | 0.8334 | 197 |
| normal | 0.8261 | 0.8265 | 0.8057 | 0.8157 | 0.8452 | 0.8985 | 0.8607 | 276 |
| sever | 0.9667 | 0.9767 | 0.9134 | 0.9393 | 0.9914 | 0.9941 | 0.9795 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=142 FP=41 TN=355 FN=55
- **normal**: TP=228 FP=55 TN=262 FN=48
- **sever**: TP=116 FP=11 TN=462 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 142 | 53 | 2 |
| normal | 39 | 228 | 9 |
| sever | 2 | 2 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7760    0.7208    0.7474       197
      normal     0.8057    0.8261    0.8157       276
       sever     0.9134    0.9667    0.9393       120

    accuracy                         0.8196       593
   macro avg     0.8317    0.8379    0.8341       593
weighted avg     0.8176    0.8196    0.8180       593

```