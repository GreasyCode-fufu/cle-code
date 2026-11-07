# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r3/convnext_tiny_seed43/convnext_tiny/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8827 |
| Accuracy | 0.8735 |
| Macro-AUROC (OvR) | 0.9797 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8732 |
| Balanced Acc | 0.8985 |
| Macro-Sensitivity | 0.8985 |
| Macro-Specificity | 0.9363 |
| Macro-Precision | 0.8763 |
| AUC-PR (Macro) | 0.9700 |
| Cohen's Kappa | 0.8039 |
| MCC | 0.8097 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 985.6 (1.01 ms/张) |
| End-to-End FPS | 282.4 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9188 | 0.8763 | 0.7870 | 0.8478 | 0.9559 | 0.9714 | 0.9532 | 197 |
| normal | 0.7935 | 0.9621 | 0.9481 | 0.8639 | 0.8425 | 0.9699 | 0.9647 | 276 |
| sever | 0.9833 | 0.9704 | 0.8939 | 0.9365 | 0.9957 | 0.9980 | 0.9922 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=181 FP=49 TN=347 FN=16
- **normal**: TP=219 FP=12 TN=305 FN=57
- **sever**: TP=118 FP=14 TN=459 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 181 | 11 | 5 |
| normal | 48 | 219 | 9 |
| sever | 1 | 1 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7870    0.9188    0.8478       197
      normal     0.9481    0.7935    0.8639       276
       sever     0.8939    0.9833    0.9365       120

    accuracy                         0.8735       593
   macro avg     0.8763    0.8985    0.8827       593
weighted avg     0.8836    0.8735    0.8732       593

```