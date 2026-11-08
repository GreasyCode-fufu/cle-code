# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16bn_dce4m_seed43/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8523 |
| Accuracy | 0.8364 |
| Macro-AUROC (OvR) | 0.9408 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8360 |
| Balanced Acc | 0.8543 |
| Macro-Sensitivity | 0.8543 |
| Macro-Specificity | 0.9095 |
| Macro-Precision | 0.8504 |
| AUC-PR (Macro) | 0.9073 |
| Cohen's Kappa | 0.7415 |
| MCC | 0.7415 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4028.9 (0.25 ms/张) |
| End-to-End FPS | 13.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7665 | 0.8914 | 0.7784 | 0.7724 | 0.8847 | 0.9143 | 0.8442 | 197 |
| normal | 0.8297 | 0.8517 | 0.8297 | 0.8297 | 0.8517 | 0.9129 | 0.8940 | 276 |
| sever | 0.9667 | 0.9852 | 0.9431 | 0.9547 | 0.9915 | 0.9953 | 0.9837 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=151 FP=43 TN=353 FN=46
- **normal**: TP=229 FP=47 TN=270 FN=47
- **sever**: TP=116 FP=7 TN=466 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 151 | 45 | 1 |
| normal | 41 | 229 | 6 |
| sever | 2 | 2 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7784    0.7665    0.7724       197
      normal     0.8297    0.8297    0.8297       276
       sever     0.9431    0.9667    0.9547       120

    accuracy                         0.8364       593
   macro avg     0.8504    0.8543    0.8523       593
weighted avg     0.8356    0.8364    0.8360       593

```