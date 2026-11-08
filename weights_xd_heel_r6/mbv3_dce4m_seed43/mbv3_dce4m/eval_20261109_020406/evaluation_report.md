# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_dce4m_seed43/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8838 |
| Accuracy | 0.8702 |
| Macro-AUROC (OvR) | 0.9660 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8699 |
| Balanced Acc | 0.8874 |
| Macro-Sensitivity | 0.8874 |
| Macro-Specificity | 0.9287 |
| Macro-Precision | 0.8804 |
| AUC-PR (Macro) | 0.9507 |
| Cohen's Kappa | 0.7953 |
| MCC | 0.7954 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3747.3 (0.27 ms/张) |
| End-to-End FPS | 18.6 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8274 | 0.9091 | 0.8191 | 0.8232 | 0.9137 | 0.9491 | 0.9241 | 197 |
| normal | 0.8514 | 0.8896 | 0.8704 | 0.8608 | 0.8731 | 0.9499 | 0.9331 | 276 |
| sever | 0.9833 | 0.9873 | 0.9516 | 0.9672 | 0.9957 | 0.9988 | 0.9951 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=163 FP=36 TN=360 FN=34
- **normal**: TP=235 FP=35 TN=282 FN=41
- **sever**: TP=118 FP=6 TN=467 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 163 | 33 | 1 |
| normal | 36 | 235 | 5 |
| sever | 0 | 2 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8191    0.8274    0.8232       197
      normal     0.8704    0.8514    0.8608       276
       sever     0.9516    0.9833    0.9672       120

    accuracy                         0.8702       593
   macro avg     0.8804    0.8874    0.8838       593
weighted avg     0.8698    0.8702    0.8699       593

```