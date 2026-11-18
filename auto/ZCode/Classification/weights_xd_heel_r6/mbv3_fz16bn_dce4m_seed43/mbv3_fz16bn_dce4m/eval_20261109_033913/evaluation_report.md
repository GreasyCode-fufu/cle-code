# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16bn_dce4m_seed43/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8587 |
| Accuracy | 0.8449 |
| Macro-AUROC (OvR) | 0.9458 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8438 |
| Balanced Acc | 0.8589 |
| Macro-Sensitivity | 0.8589 |
| Macro-Specificity | 0.9128 |
| Macro-Precision | 0.8597 |
| AUC-PR (Macro) | 0.9175 |
| Cohen's Kappa | 0.7537 |
| MCC | 0.7544 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4221.1 (0.24 ms/张) |
| End-to-End FPS | 11.9 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7513 | 0.9141 | 0.8132 | 0.7810 | 0.8808 | 0.9188 | 0.8715 | 197 |
| normal | 0.8587 | 0.8391 | 0.8229 | 0.8404 | 0.8721 | 0.9202 | 0.8868 | 276 |
| sever | 0.9667 | 0.9852 | 0.9431 | 0.9547 | 0.9915 | 0.9985 | 0.9944 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=148 FP=34 TN=362 FN=49
- **normal**: TP=237 FP=51 TN=266 FN=39
- **sever**: TP=116 FP=7 TN=466 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 148 | 47 | 2 |
| normal | 34 | 237 | 5 |
| sever | 0 | 4 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8132    0.7513    0.7810       197
      normal     0.8229    0.8587    0.8404       276
       sever     0.9431    0.9667    0.9547       120

    accuracy                         0.8449       593
   macro avg     0.8597    0.8589    0.8587       593
weighted avg     0.8440    0.8449    0.8438       593

```