# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/convnext_v2c_seed44/convnext_v2c/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8923 |
| Accuracy | 0.8904 |
| Macro-AUROC (OvR) | 0.9783 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8900 |
| Balanced Acc | 0.9053 |
| Macro-Sensitivity | 0.9053 |
| Macro-Specificity | 0.9435 |
| Macro-Precision | 0.8834 |
| AUC-PR (Macro) | 0.9671 |
| Cohen's Kappa | 0.8286 |
| MCC | 0.8297 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1654.6 (0.60 ms/张) |
| End-to-End FPS | 322.7 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8629 | 0.9495 | 0.8947 | 0.8786 | 0.9330 | 0.9714 | 0.9552 | 197 |
| normal | 0.8696 | 0.9274 | 0.9125 | 0.8905 | 0.8909 | 0.9671 | 0.9611 | 276 |
| sever | 0.9833 | 0.9535 | 0.8429 | 0.9077 | 0.9956 | 0.9963 | 0.9849 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=170 FP=20 TN=376 FN=27
- **normal**: TP=240 FP=23 TN=294 FN=36
- **sever**: TP=118 FP=22 TN=451 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 170 | 22 | 5 |
| normal | 19 | 240 | 17 |
| sever | 1 | 1 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8947    0.8629    0.8786       197
      normal     0.9125    0.8696    0.8905       276
       sever     0.8429    0.9833    0.9077       120

    accuracy                         0.8904       593
   macro avg     0.8834    0.9053    0.8923       593
weighted avg     0.8925    0.8904    0.8900       593

```