# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone/convnext_tiny_seed42/convnext_tiny/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8952 |
| Accuracy | 0.8887 |
| Macro-AUROC (OvR) | 0.9773 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8879 |
| Balanced Acc | 0.9032 |
| Macro-Sensitivity | 0.9032 |
| Macro-Specificity | 0.9404 |
| Macro-Precision | 0.8889 |
| AUC-PR (Macro) | 0.9675 |
| Cohen's Kappa | 0.8250 |
| MCC | 0.8254 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 675.6 (1.48 ms/张) |
| End-to-End FPS | 248.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8376 | 0.9369 | 0.8684 | 0.8527 | 0.9206 | 0.9654 | 0.9446 | 197 |
| normal | 0.8804 | 0.9117 | 0.8967 | 0.8885 | 0.8975 | 0.9676 | 0.9622 | 276 |
| sever | 0.9917 | 0.9725 | 0.9015 | 0.9444 | 0.9978 | 0.9989 | 0.9957 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=165 FP=25 TN=371 FN=32
- **normal**: TP=243 FP=28 TN=289 FN=33
- **sever**: TP=119 FP=13 TN=460 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 165 | 27 | 5 |
| normal | 25 | 243 | 8 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8684    0.8376    0.8527       197
      normal     0.8967    0.8804    0.8885       276
       sever     0.9015    0.9917    0.9444       120

    accuracy                         0.8887       593
   macro avg     0.8889    0.9032    0.8952       593
weighted avg     0.8883    0.8887    0.8879       593

```