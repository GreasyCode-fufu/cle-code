# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce16_seed43/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8653 |
| Accuracy | 0.8516 |
| Macro-AUROC (OvR) | 0.9438 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8519 |
| Balanced Acc | 0.8608 |
| Macro-Sensitivity | 0.8608 |
| Macro-Specificity | 0.9167 |
| Macro-Precision | 0.8703 |
| AUC-PR (Macro) | 0.9150 |
| Cohen's Kappa | 0.7641 |
| MCC | 0.7643 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 3333.8 (0.30 ms/张) |
| End-to-End FPS | 17.6 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7868 | 0.9015 | 0.7990 | 0.7928 | 0.8947 | 0.9147 | 0.8634 | 197 |
| normal | 0.8623 | 0.8549 | 0.8380 | 0.8500 | 0.8770 | 0.9187 | 0.8898 | 276 |
| sever | 0.9333 | 0.9937 | 0.9739 | 0.9532 | 0.9833 | 0.9980 | 0.9920 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=155 FP=39 TN=357 FN=42
- **normal**: TP=238 FP=46 TN=271 FN=38
- **sever**: TP=112 FP=3 TN=470 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 155 | 42 | 0 |
| normal | 35 | 238 | 3 |
| sever | 4 | 4 | 112 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7990    0.7868    0.7928       197
      normal     0.8380    0.8623    0.8500       276
       sever     0.9739    0.9333    0.9532       120

    accuracy                         0.8516       593
   macro avg     0.8703    0.8608    0.8653       593
weighted avg     0.8526    0.8516    0.8519       593

```