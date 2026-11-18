# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_r6_fz16bn_seed43/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7805 |
| Accuracy | 0.7622 |
| Macro-AUROC (OvR) | 0.9124 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7600 |
| Balanced Acc | 0.8032 |
| Macro-Sensitivity | 0.8032 |
| Macro-Specificity | 0.8768 |
| Macro-Precision | 0.7710 |
| AUC-PR (Macro) | 0.8621 |
| Cohen's Kappa | 0.6338 |
| MCC | 0.6403 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3537.8 (0.28 ms/张) |
| End-to-End FPS | 160.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7716 | 0.8030 | 0.6609 | 0.7119 | 0.8760 | 0.8698 | 0.7695 | 197 |
| normal | 0.6630 | 0.8801 | 0.8281 | 0.7364 | 0.7500 | 0.8725 | 0.8356 | 276 |
| sever | 0.9750 | 0.9471 | 0.8239 | 0.8931 | 0.9933 | 0.9947 | 0.9812 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=152 FP=78 TN=318 FN=45
- **normal**: TP=183 FP=38 TN=279 FN=93
- **sever**: TP=117 FP=25 TN=448 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 152 | 36 | 9 |
| normal | 77 | 183 | 16 |
| sever | 1 | 2 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6609    0.7716    0.7119       197
      normal     0.8281    0.6630    0.7364       276
       sever     0.8239    0.9750    0.8931       120

    accuracy                         0.7622       593
   macro avg     0.7710    0.8032    0.7805       593
weighted avg     0.7717    0.7622    0.7600       593

```