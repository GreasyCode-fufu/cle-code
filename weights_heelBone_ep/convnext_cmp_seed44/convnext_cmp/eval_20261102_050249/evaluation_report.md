# convnext_cmp 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_cmp_seed44/convnext_cmp/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8474 |
| Accuracy | 0.8432 |
| Macro-AUROC (OvR) | 0.9779 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8413 |
| Balanced Acc | 0.8794 |
| Macro-Sensitivity | 0.8794 |
| Macro-Specificity | 0.9248 |
| Macro-Precision | 0.8394 |
| AUC-PR (Macro) | 0.9660 |
| Cohen's Kappa | 0.7606 |
| MCC | 0.7724 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 863.7 (1.16 ms/张) |
| End-to-End FPS | 275.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9137 | 0.8737 | 0.7826 | 0.8431 | 0.9532 | 0.9673 | 0.9430 | 197 |
| normal | 0.7246 | 0.9748 | 0.9615 | 0.8264 | 0.8026 | 0.9684 | 0.9644 | 276 |
| sever | 1.0000 | 0.9260 | 0.7742 | 0.8727 | 1.0000 | 0.9979 | 0.9906 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=180 FP=50 TN=346 FN=17
- **normal**: TP=200 FP=8 TN=309 FN=76
- **sever**: TP=120 FP=35 TN=438 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 180 | 8 | 9 |
| normal | 50 | 200 | 26 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7826    0.9137    0.8431       197
      normal     0.9615    0.7246    0.8264       276
       sever     0.7742    1.0000    0.8727       120

    accuracy                         0.8432       593
   macro avg     0.8394    0.8794    0.8474       593
weighted avg     0.8642    0.8432    0.8413       593

```