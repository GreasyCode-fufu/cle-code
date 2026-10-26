# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/resnet50_seed43/resnet50/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9063 |
| Accuracy | 0.9039 |
| Macro-AUROC (OvR) | 0.9774 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9031 |
| Balanced Acc | 0.9016 |
| Macro-Sensitivity | 0.9016 |
| Macro-Specificity | 0.9454 |
| Macro-Precision | 0.9136 |
| AUC-PR (Macro) | 0.9670 |
| Cohen's Kappa | 0.8467 |
| MCC | 0.8485 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 1044.1 (0.96 ms/张) |
| End-to-End FPS | 285.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8223 | 0.9646 | 0.9205 | 0.8686 | 0.9161 | 0.9630 | 0.9454 | 197 |
| normal | 0.9493 | 0.8864 | 0.8792 | 0.9129 | 0.9525 | 0.9736 | 0.9684 | 276 |
| sever | 0.9333 | 0.9852 | 0.9412 | 0.9372 | 0.9831 | 0.9958 | 0.9873 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=162 FP=14 TN=382 FN=35
- **normal**: TP=262 FP=36 TN=281 FN=14
- **sever**: TP=112 FP=7 TN=466 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 162 | 30 | 5 |
| normal | 12 | 262 | 2 |
| sever | 2 | 6 | 112 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.9205    0.8223    0.8686       197
      normal     0.8792    0.9493    0.9129       276
       sever     0.9412    0.9333    0.9372       120

    accuracy                         0.9039       593
   macro avg     0.9136    0.9016    0.9063       593
weighted avg     0.9054    0.9039    0.9031       593

```