# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/convnext_v2bc_seed44/convnext_v2bc/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8909 |
| Accuracy | 0.8853 |
| Macro-AUROC (OvR) | 0.9774 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8845 |
| Balanced Acc | 0.9003 |
| Macro-Sensitivity | 0.9003 |
| Macro-Specificity | 0.9392 |
| Macro-Precision | 0.8836 |
| AUC-PR (Macro) | 0.9682 |
| Cohen's Kappa | 0.8199 |
| MCC | 0.8205 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 797.5 (1.25 ms/张) |
| End-to-End FPS | 267.0 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8325 | 0.9343 | 0.8632 | 0.8475 | 0.9181 | 0.9636 | 0.9448 | 197 |
| normal | 0.8768 | 0.9148 | 0.8996 | 0.8881 | 0.8951 | 0.9696 | 0.9629 | 276 |
| sever | 0.9917 | 0.9683 | 0.8881 | 0.9370 | 0.9978 | 0.9992 | 0.9968 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=164 FP=26 TN=370 FN=33
- **normal**: TP=242 FP=27 TN=290 FN=34
- **sever**: TP=119 FP=15 TN=458 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 164 | 26 | 7 |
| normal | 26 | 242 | 8 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8632    0.8325    0.8475       197
      normal     0.8996    0.8768    0.8881       276
       sever     0.8881    0.9917    0.9370       120

    accuracy                         0.8853       593
   macro avg     0.8836    0.9003    0.8909       593
weighted avg     0.8852    0.8853    0.8845       593

```