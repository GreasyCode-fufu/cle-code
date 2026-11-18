# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_svd_seed44/convnext_svd/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8784 |
| Accuracy | 0.8718 |
| Macro-AUROC (OvR) | 0.9762 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8715 |
| Balanced Acc | 0.8927 |
| Macro-Sensitivity | 0.8927 |
| Macro-Specificity | 0.9348 |
| Macro-Precision | 0.8701 |
| AUC-PR (Macro) | 0.9657 |
| Cohen's Kappa | 0.8007 |
| MCC | 0.8043 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 940.4 (1.06 ms/张) |
| End-to-End FPS | 234.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9036 | 0.8939 | 0.8091 | 0.8537 | 0.9491 | 0.9644 | 0.9482 | 197 |
| normal | 0.8080 | 0.9464 | 0.9292 | 0.8643 | 0.8499 | 0.9690 | 0.9629 | 276 |
| sever | 0.9667 | 0.9641 | 0.8722 | 0.9170 | 0.9913 | 0.9952 | 0.9860 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=178 FP=42 TN=354 FN=19
- **normal**: TP=223 FP=17 TN=300 FN=53
- **sever**: TP=116 FP=17 TN=456 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 178 | 15 | 4 |
| normal | 40 | 223 | 13 |
| sever | 2 | 2 | 116 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8091    0.9036    0.8537       197
      normal     0.9292    0.8080    0.8643       276
       sever     0.8722    0.9667    0.9170       120

    accuracy                         0.8718       593
   macro avg     0.8701    0.8927    0.8784       593
weighted avg     0.8777    0.8718    0.8715       593

```