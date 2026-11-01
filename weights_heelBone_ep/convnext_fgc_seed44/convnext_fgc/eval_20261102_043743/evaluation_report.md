# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_fgc_seed44/convnext_fgc/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8972 |
| Accuracy | 0.8904 |
| Macro-AUROC (OvR) | 0.9785 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8899 |
| Balanced Acc | 0.9094 |
| Macro-Sensitivity | 0.9094 |
| Macro-Specificity | 0.9432 |
| Macro-Precision | 0.8880 |
| AUC-PR (Macro) | 0.9692 |
| Cohen's Kappa | 0.8287 |
| MCC | 0.8300 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 1112.4 (0.90 ms/张) |
| End-to-End FPS | 290.0 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8731 | 0.9192 | 0.8431 | 0.8579 | 0.9357 | 0.9641 | 0.9419 | 197 |
| normal | 0.8551 | 0.9401 | 0.9255 | 0.8889 | 0.8817 | 0.9719 | 0.9671 | 276 |
| sever | 1.0000 | 0.9704 | 0.8955 | 0.9449 | 1.0000 | 0.9996 | 0.9984 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=172 FP=32 TN=364 FN=25
- **normal**: TP=236 FP=19 TN=298 FN=40
- **sever**: TP=120 FP=14 TN=459 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 172 | 19 | 6 |
| normal | 32 | 236 | 8 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8431    0.8731    0.8579       197
      normal     0.9255    0.8551    0.8889       276
       sever     0.8955    1.0000    0.9449       120

    accuracy                         0.8904       593
   macro avg     0.8880    0.9094    0.8972       593
weighted avg     0.8921    0.8904    0.8899       593

```