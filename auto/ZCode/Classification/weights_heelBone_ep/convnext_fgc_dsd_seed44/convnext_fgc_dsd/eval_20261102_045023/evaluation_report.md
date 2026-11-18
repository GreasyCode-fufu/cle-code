# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_fgc_dsd_seed44/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8376 |
| Accuracy | 0.8398 |
| Macro-AUROC (OvR) | 0.9665 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8403 |
| Balanced Acc | 0.8634 |
| Macro-Sensitivity | 0.8634 |
| Macro-Specificity | 0.9209 |
| Macro-Precision | 0.8302 |
| AUC-PR (Macro) | 0.9505 |
| Cohen's Kappa | 0.7528 |
| MCC | 0.7584 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 1012.6 (0.99 ms/张) |
| End-to-End FPS | 284.2 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7868 | 0.9495 | 0.8857 | 0.8333 | 0.8995 | 0.9551 | 0.9305 | 197 |
| normal | 0.8116 | 0.9148 | 0.8924 | 0.8501 | 0.8480 | 0.9513 | 0.9454 | 276 |
| sever | 0.9917 | 0.8985 | 0.7126 | 0.8293 | 0.9977 | 0.9931 | 0.9758 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=155 FP=20 TN=376 FN=42
- **normal**: TP=224 FP=27 TN=290 FN=52
- **sever**: TP=119 FP=48 TN=425 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 155 | 26 | 16 |
| normal | 20 | 224 | 32 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8857    0.7868    0.8333       197
      normal     0.8924    0.8116    0.8501       276
       sever     0.7126    0.9917    0.8293       120

    accuracy                         0.8398       593
   macro avg     0.8302    0.8634    0.8376       593
weighted avg     0.8538    0.8398    0.8403       593

```