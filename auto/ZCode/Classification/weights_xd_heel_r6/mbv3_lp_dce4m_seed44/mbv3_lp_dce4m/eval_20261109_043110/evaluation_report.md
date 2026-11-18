# mbv3_lp_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_lp_dce4m_seed44/mbv3_lp_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7881 |
| Accuracy | 0.7707 |
| Macro-AUROC (OvR) | 0.8967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7686 |
| Balanced Acc | 0.7923 |
| Macro-Sensitivity | 0.7923 |
| Macro-Specificity | 0.8725 |
| Macro-Precision | 0.7858 |
| AUC-PR (Macro) | 0.8346 |
| Cohen's Kappa | 0.6369 |
| MCC | 0.6376 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 2338.9 (0.43 ms/张) |
| End-to-End FPS | 16.0 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.6599 | 0.8763 | 0.7263 | 0.6915 | 0.8382 | 0.8610 | 0.7489 | 197 |
| normal | 0.7754 | 0.7729 | 0.7483 | 0.7616 | 0.7980 | 0.8389 | 0.7995 | 276 |
| sever | 0.9417 | 0.9683 | 0.8828 | 0.9113 | 0.9849 | 0.9902 | 0.9554 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=130 FP=49 TN=347 FN=67
- **normal**: TP=214 FP=72 TN=245 FN=62
- **sever**: TP=113 FP=15 TN=458 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 130 | 65 | 2 |
| normal | 49 | 214 | 13 |
| sever | 0 | 7 | 113 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7263    0.6599    0.6915       197
      normal     0.7483    0.7754    0.7616       276
       sever     0.8828    0.9417    0.9113       120

    accuracy                         0.7707       593
   macro avg     0.7858    0.7923    0.7881       593
weighted avg     0.7682    0.7707    0.7686       593

```