# deit_small_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/deit_small_patch16_224_seed43/deit_small_patch16_224/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8168 |
| Accuracy | 0.8111 |
| Macro-AUROC (OvR) | 0.9449 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8087 |
| Balanced Acc | 0.8395 |
| Macro-Sensitivity | 0.8395 |
| Macro-Specificity | 0.9016 |
| Macro-Precision | 0.8059 |
| AUC-PR (Macro) | 0.9221 |
| Cohen's Kappa | 0.7066 |
| MCC | 0.7097 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.25G |
| FLOPs | 8.50G |
| Params | 21.59M |
| Pure-Inference FPS | 938.8 (1.07 ms/张) |
| End-to-End FPS | 273.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7360 | 0.9167 | 0.8146 | 0.7733 | 0.8747 | 0.9209 | 0.8862 | 197 |
| normal | 0.7826 | 0.8644 | 0.8340 | 0.8075 | 0.8204 | 0.9171 | 0.8927 | 276 |
| sever | 1.0000 | 0.9239 | 0.7692 | 0.8696 | 1.0000 | 0.9967 | 0.9874 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=145 FP=33 TN=363 FN=52
- **normal**: TP=216 FP=43 TN=274 FN=60
- **sever**: TP=120 FP=36 TN=437 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 145 | 43 | 9 |
| normal | 33 | 216 | 27 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8146    0.7360    0.7733       197
      normal     0.8340    0.7826    0.8075       276
       sever     0.7692    1.0000    0.8696       120

    accuracy                         0.8111       593
   macro avg     0.8059    0.8395    0.8168       593
weighted avg     0.8144    0.8111    0.8087       593

```