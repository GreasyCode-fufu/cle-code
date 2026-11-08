# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16bn_dce4m_seed44/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8382 |
| Accuracy | 0.8196 |
| Macro-AUROC (OvR) | 0.9333 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8190 |
| Balanced Acc | 0.8403 |
| Macro-Sensitivity | 0.8403 |
| Macro-Specificity | 0.8998 |
| Macro-Precision | 0.8363 |
| AUC-PR (Macro) | 0.8972 |
| Cohen's Kappa | 0.7148 |
| MCC | 0.7148 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4491.0 (0.22 ms/张) |
| End-to-End FPS | 13.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7462 | 0.8813 | 0.7577 | 0.7519 | 0.8747 | 0.9010 | 0.8423 | 197 |
| normal | 0.8080 | 0.8328 | 0.8080 | 0.8080 | 0.8328 | 0.9013 | 0.8571 | 276 |
| sever | 0.9667 | 0.9852 | 0.9431 | 0.9547 | 0.9915 | 0.9976 | 0.9921 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=147 FP=47 TN=349 FN=50
- **normal**: TP=223 FP=53 TN=264 FN=53
- **sever**: TP=116 FP=7 TN=466 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 147 | 49 | 1 |
| normal | 47 | 223 | 6 |
| sever | 0 | 4 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7577    0.7462    0.7519       197
      normal     0.8080    0.8080    0.8080       276
       sever     0.9431    0.9667    0.9547       120

    accuracy                         0.8196       593
   macro avg     0.8363    0.8403    0.8382       593
weighted avg     0.8186    0.8196    0.8190       593

```