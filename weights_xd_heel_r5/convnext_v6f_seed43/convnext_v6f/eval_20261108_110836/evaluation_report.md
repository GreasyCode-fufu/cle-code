# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r5/convnext_v6f_seed43/convnext_v6f/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9128 |
| Accuracy | 0.9022 |
| Macro-AUROC (OvR) | 0.9784 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9023 |
| Balanced Acc | 0.9162 |
| Macro-Sensitivity | 0.9162 |
| Macro-Specificity | 0.9469 |
| Macro-Precision | 0.9100 |
| AUC-PR (Macro) | 0.9673 |
| Cohen's Kappa | 0.8460 |
| MCC | 0.8464 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 1629.2 (0.61 ms/张) |
| End-to-End FPS | 323.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8883 | 0.9217 | 0.8495 | 0.8685 | 0.9432 | 0.9705 | 0.9521 | 197 |
| normal | 0.8768 | 0.9274 | 0.9132 | 0.8946 | 0.8963 | 0.9671 | 0.9609 | 276 |
| sever | 0.9833 | 0.9915 | 0.9672 | 0.9752 | 0.9958 | 0.9976 | 0.9890 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=175 FP=31 TN=365 FN=22
- **normal**: TP=242 FP=23 TN=294 FN=34
- **sever**: TP=118 FP=4 TN=469 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 175 | 21 | 1 |
| normal | 31 | 242 | 3 |
| sever | 0 | 2 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8495    0.8883    0.8685       197
      normal     0.9132    0.8768    0.8946       276
       sever     0.9672    0.9833    0.9752       120

    accuracy                         0.9022       593
   macro avg     0.9100    0.9162    0.9128       593
weighted avg     0.9030    0.9022    0.9023       593

```