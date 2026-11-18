# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz8_dce4m_seed43/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8650 |
| Accuracy | 0.8516 |
| Macro-AUROC (OvR) | 0.9551 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8518 |
| Balanced Acc | 0.8623 |
| Macro-Sensitivity | 0.8623 |
| Macro-Specificity | 0.9170 |
| Macro-Precision | 0.8679 |
| AUC-PR (Macro) | 0.9329 |
| Cohen's Kappa | 0.7646 |
| MCC | 0.7646 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4057.7 (0.25 ms/张) |
| End-to-End FPS | 13.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8020 | 0.9066 | 0.8103 | 0.8061 | 0.9020 | 0.9377 | 0.9039 | 197 |
| normal | 0.8514 | 0.8549 | 0.8363 | 0.8438 | 0.8686 | 0.9325 | 0.9097 | 276 |
| sever | 0.9333 | 0.9894 | 0.9573 | 0.9451 | 0.9832 | 0.9951 | 0.9851 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=158 FP=37 TN=359 FN=39
- **normal**: TP=235 FP=46 TN=271 FN=41
- **sever**: TP=112 FP=5 TN=468 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 158 | 39 | 0 |
| normal | 36 | 235 | 5 |
| sever | 1 | 7 | 112 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8103    0.8020    0.8061       197
      normal     0.8363    0.8514    0.8438       276
       sever     0.9573    0.9333    0.9451       120

    accuracy                         0.8516       593
   macro avg     0.8679    0.8623    0.8650       593
weighted avg     0.8521    0.8516    0.8518       593

```