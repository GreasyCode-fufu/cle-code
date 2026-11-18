# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16bn_dce4m_seed42/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8450 |
| Accuracy | 0.8331 |
| Macro-AUROC (OvR) | 0.9357 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8321 |
| Balanced Acc | 0.8473 |
| Macro-Sensitivity | 0.8473 |
| Macro-Specificity | 0.9077 |
| Macro-Precision | 0.8435 |
| AUC-PR (Macro) | 0.8992 |
| Cohen's Kappa | 0.7358 |
| MCC | 0.7361 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3892.0 (0.26 ms/张) |
| End-to-End FPS | 9.9 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7513 | 0.9040 | 0.7957 | 0.7728 | 0.8796 | 0.9073 | 0.8484 | 197 |
| normal | 0.8406 | 0.8423 | 0.8227 | 0.8315 | 0.8585 | 0.9065 | 0.8768 | 276 |
| sever | 0.9500 | 0.9767 | 0.9120 | 0.9306 | 0.9872 | 0.9933 | 0.9723 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=148 FP=38 TN=358 FN=49
- **normal**: TP=232 FP=50 TN=267 FN=44
- **sever**: TP=114 FP=11 TN=462 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 148 | 46 | 3 |
| normal | 36 | 232 | 8 |
| sever | 2 | 4 | 114 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7957    0.7513    0.7728       197
      normal     0.8227    0.8406    0.8315       276
       sever     0.9120    0.9500    0.9306       120

    accuracy                         0.8331       593
   macro avg     0.8435    0.8473    0.8450       593
weighted avg     0.8318    0.8331    0.8321       593

```