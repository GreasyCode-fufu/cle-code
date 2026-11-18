# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce4s_seed43/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8504 |
| Accuracy | 0.8364 |
| Macro-AUROC (OvR) | 0.9433 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8355 |
| Balanced Acc | 0.8503 |
| Macro-Sensitivity | 0.8503 |
| Macro-Specificity | 0.9086 |
| Macro-Precision | 0.8512 |
| AUC-PR (Macro) | 0.9106 |
| Cohen's Kappa | 0.7405 |
| MCC | 0.7409 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3626.1 (0.28 ms/张) |
| End-to-End FPS | 18.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7411 | 0.9015 | 0.7892 | 0.7644 | 0.8750 | 0.9169 | 0.8479 | 197 |
| normal | 0.8514 | 0.8391 | 0.8217 | 0.8363 | 0.8664 | 0.9171 | 0.8999 | 276 |
| sever | 0.9583 | 0.9852 | 0.9426 | 0.9504 | 0.9894 | 0.9960 | 0.9840 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=146 FP=39 TN=357 FN=51
- **normal**: TP=235 FP=51 TN=266 FN=41
- **sever**: TP=115 FP=7 TN=466 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 146 | 49 | 2 |
| normal | 36 | 235 | 5 |
| sever | 3 | 2 | 115 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7892    0.7411    0.7644       197
      normal     0.8217    0.8514    0.8363       276
       sever     0.9426    0.9583    0.9504       120

    accuracy                         0.8364       593
   macro avg     0.8512    0.8503    0.8504       593
weighted avg     0.8354    0.8364    0.8355       593

```