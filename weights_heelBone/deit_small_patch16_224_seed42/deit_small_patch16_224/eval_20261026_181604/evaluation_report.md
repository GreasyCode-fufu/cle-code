# deit_small_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/deit_small_patch16_224_seed42/deit_small_patch16_224/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8295 |
| Accuracy | 0.8162 |
| Macro-AUROC (OvR) | 0.9479 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8156 |
| Balanced Acc | 0.8463 |
| Macro-Sensitivity | 0.8463 |
| Macro-Specificity | 0.9059 |
| Macro-Precision | 0.8252 |
| AUC-PR (Macro) | 0.9215 |
| Cohen's Kappa | 0.7158 |
| MCC | 0.7236 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.25G |
| FLOPs | 8.50G |
| Params | 21.59M |
| Pure-Inference FPS | 1306.4 (0.77 ms/张) |
| End-to-End FPS | 300.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8680 | 0.8283 | 0.7155 | 0.7844 | 0.9266 | 0.9254 | 0.8789 | 197 |
| normal | 0.7210 | 0.9274 | 0.8964 | 0.7992 | 0.7925 | 0.9242 | 0.9025 | 276 |
| sever | 0.9500 | 0.9619 | 0.8636 | 0.9048 | 0.9870 | 0.9942 | 0.9831 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=171 FP=68 TN=328 FN=26
- **normal**: TP=199 FP=23 TN=294 FN=77
- **sever**: TP=114 FP=18 TN=455 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 171 | 21 | 5 |
| normal | 64 | 199 | 13 |
| sever | 4 | 2 | 114 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7155    0.8680    0.7844       197
      normal     0.8964    0.7210    0.7992       276
       sever     0.8636    0.9500    0.9048       120

    accuracy                         0.8162       593
   macro avg     0.8252    0.8463    0.8295       593
weighted avg     0.8297    0.8162    0.8156       593

```