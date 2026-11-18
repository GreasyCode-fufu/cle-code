# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_fz16_dce4s_seed44/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8545 |
| Accuracy | 0.8381 |
| Macro-AUROC (OvR) | 0.9455 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8379 |
| Balanced Acc | 0.8530 |
| Macro-Sensitivity | 0.8530 |
| Macro-Specificity | 0.9095 |
| Macro-Precision | 0.8562 |
| AUC-PR (Macro) | 0.9178 |
| Cohen's Kappa | 0.7432 |
| MCC | 0.7433 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3904.4 (0.26 ms/张) |
| End-to-End FPS | 18.9 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7563 | 0.8914 | 0.7760 | 0.7661 | 0.8803 | 0.9209 | 0.8583 | 197 |
| normal | 0.8442 | 0.8454 | 0.8262 | 0.8351 | 0.8617 | 0.9189 | 0.9054 | 276 |
| sever | 0.9583 | 0.9915 | 0.9664 | 0.9623 | 0.9895 | 0.9968 | 0.9898 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=149 FP=43 TN=353 FN=48
- **normal**: TP=233 FP=49 TN=268 FN=43
- **sever**: TP=115 FP=4 TN=469 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 149 | 47 | 1 |
| normal | 40 | 233 | 3 |
| sever | 3 | 2 | 115 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7760    0.7563    0.7661       197
      normal     0.8262    0.8442    0.8351       276
       sever     0.9664    0.9583    0.9623       120

    accuracy                         0.8381       593
   macro avg     0.8562    0.8530    0.8545       593
weighted avg     0.8379    0.8381    0.8379       593

```