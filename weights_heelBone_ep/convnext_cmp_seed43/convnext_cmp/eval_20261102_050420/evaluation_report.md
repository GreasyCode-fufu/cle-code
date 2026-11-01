# convnext_cmp 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_cmp_seed43/convnext_cmp/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8980 |
| Accuracy | 0.8870 |
| Macro-AUROC (OvR) | 0.9804 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8868 |
| Balanced Acc | 0.9064 |
| Macro-Sensitivity | 0.9064 |
| Macro-Specificity | 0.9401 |
| Macro-Precision | 0.8918 |
| AUC-PR (Macro) | 0.9713 |
| Cohen's Kappa | 0.8230 |
| MCC | 0.8243 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 1688.2 (0.59 ms/张) |
| End-to-End FPS | 325.9 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8832 | 0.9066 | 0.8246 | 0.8529 | 0.9398 | 0.9706 | 0.9494 | 197 |
| normal | 0.8442 | 0.9306 | 0.9137 | 0.8776 | 0.8728 | 0.9713 | 0.9675 | 276 |
| sever | 0.9917 | 0.9831 | 0.9370 | 0.9636 | 0.9979 | 0.9992 | 0.9969 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=174 FP=37 TN=359 FN=23
- **normal**: TP=233 FP=22 TN=295 FN=43
- **sever**: TP=119 FP=8 TN=465 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 174 | 21 | 2 |
| normal | 37 | 233 | 6 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8246    0.8832    0.8529       197
      normal     0.9137    0.8442    0.8776       276
       sever     0.9370    0.9917    0.9636       120

    accuracy                         0.8870       593
   macro avg     0.8918    0.9064    0.8980       593
weighted avg     0.8888    0.8870    0.8868       593

```