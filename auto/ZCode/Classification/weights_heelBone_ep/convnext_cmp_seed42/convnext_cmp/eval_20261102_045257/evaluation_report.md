# convnext_cmp 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_cmp_seed42/convnext_cmp/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8722 |
| Accuracy | 0.8634 |
| Macro-AUROC (OvR) | 0.9777 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8625 |
| Balanced Acc | 0.8915 |
| Macro-Sensitivity | 0.8915 |
| Macro-Specificity | 0.9311 |
| Macro-Precision | 0.8622 |
| AUC-PR (Macro) | 0.9684 |
| Cohen's Kappa | 0.7885 |
| MCC | 0.7935 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 905.7 (1.10 ms/张) |
| End-to-End FPS | 275.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8883 | 0.8838 | 0.7919 | 0.8373 | 0.9409 | 0.9635 | 0.9439 | 197 |
| normal | 0.7862 | 0.9495 | 0.9313 | 0.8527 | 0.8361 | 0.9701 | 0.9631 | 276 |
| sever | 1.0000 | 0.9598 | 0.8633 | 0.9266 | 1.0000 | 0.9996 | 0.9983 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=175 FP=46 TN=350 FN=22
- **normal**: TP=217 FP=16 TN=301 FN=59
- **sever**: TP=120 FP=19 TN=454 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 175 | 16 | 6 |
| normal | 46 | 217 | 13 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7919    0.8883    0.8373       197
      normal     0.9313    0.7862    0.8527       276
       sever     0.8633    1.0000    0.9266       120

    accuracy                         0.8634       593
   macro avg     0.8622    0.8915    0.8722       593
weighted avg     0.8712    0.8634    0.8625       593

```