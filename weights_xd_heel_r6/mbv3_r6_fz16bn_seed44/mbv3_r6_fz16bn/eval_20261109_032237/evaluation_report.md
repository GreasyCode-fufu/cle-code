# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz16bn_seed44/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7721 |
| Accuracy | 0.7555 |
| Macro-AUROC (OvR) | 0.9141 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7523 |
| Balanced Acc | 0.7984 |
| Macro-Sensitivity | 0.7984 |
| Macro-Specificity | 0.8739 |
| Macro-Precision | 0.7612 |
| AUC-PR (Macro) | 0.8645 |
| Cohen's Kappa | 0.6244 |
| MCC | 0.6315 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3875.5 (0.26 ms/张) |
| End-to-End FPS | 130.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7716 | 0.8081 | 0.6667 | 0.7153 | 0.8767 | 0.8748 | 0.7749 | 197 |
| normal | 0.6486 | 0.8770 | 0.8211 | 0.7247 | 0.7413 | 0.8763 | 0.8415 | 276 |
| sever | 0.9750 | 0.9366 | 0.7959 | 0.8764 | 0.9933 | 0.9912 | 0.9771 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=152 FP=76 TN=320 FN=45
- **normal**: TP=179 FP=39 TN=278 FN=97
- **sever**: TP=117 FP=30 TN=443 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 152 | 37 | 8 |
| normal | 75 | 179 | 22 |
| sever | 1 | 2 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6667    0.7716    0.7153       197
      normal     0.8211    0.6486    0.7247       276
       sever     0.7959    0.9750    0.8764       120

    accuracy                         0.7555       593
   macro avg     0.7612    0.7984    0.7721       593
weighted avg     0.7647    0.7555    0.7523       593

```