# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_abl/convnext_v2m_seed44/convnext_v2m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9192 |
| Accuracy | 0.9123 |
| Macro-AUROC (OvR) | 0.9818 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9117 |
| Balanced Acc | 0.9170 |
| Macro-Sensitivity | 0.9170 |
| Macro-Specificity | 0.9501 |
| Macro-Precision | 0.9229 |
| AUC-PR (Macro) | 0.9735 |
| Cohen's Kappa | 0.8606 |
| MCC | 0.8615 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 893.1 (1.12 ms/张) |
| End-to-End FPS | 277.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8376 | 0.9596 | 0.9116 | 0.8730 | 0.9223 | 0.9716 | 0.9540 | 197 |
| normal | 0.9384 | 0.8991 | 0.8900 | 0.9136 | 0.9437 | 0.9745 | 0.9685 | 276 |
| sever | 0.9750 | 0.9915 | 0.9669 | 0.9710 | 0.9936 | 0.9994 | 0.9979 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=165 FP=16 TN=380 FN=32
- **normal**: TP=259 FP=32 TN=285 FN=17
- **sever**: TP=117 FP=4 TN=469 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 165 | 29 | 3 |
| normal | 16 | 259 | 1 |
| sever | 0 | 3 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.9116    0.8376    0.8730       197
      normal     0.8900    0.9384    0.9136       276
       sever     0.9669    0.9750    0.9710       120

    accuracy                         0.9123       593
   macro avg     0.9229    0.9170    0.9192       593
weighted avg     0.9128    0.9123    0.9117       593

```