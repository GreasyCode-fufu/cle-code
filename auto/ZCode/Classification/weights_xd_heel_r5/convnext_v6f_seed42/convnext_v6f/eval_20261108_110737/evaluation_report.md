# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r5/convnext_v6f_seed42/convnext_v6f/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8913 |
| Accuracy | 0.8836 |
| Macro-AUROC (OvR) | 0.9764 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8830 |
| Balanced Acc | 0.8985 |
| Macro-Sensitivity | 0.8985 |
| Macro-Specificity | 0.9375 |
| Macro-Precision | 0.8852 |
| AUC-PR (Macro) | 0.9654 |
| Cohen's Kappa | 0.8170 |
| MCC | 0.8173 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 641.1 (1.56 ms/张) |
| End-to-End FPS | 243.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8426 | 0.9293 | 0.8557 | 0.8491 | 0.9223 | 0.9647 | 0.9447 | 197 |
| normal | 0.8696 | 0.9085 | 0.8922 | 0.8807 | 0.8889 | 0.9659 | 0.9566 | 276 |
| sever | 0.9833 | 0.9746 | 0.9077 | 0.9440 | 0.9957 | 0.9987 | 0.9948 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=166 FP=28 TN=368 FN=31
- **normal**: TP=240 FP=29 TN=288 FN=36
- **sever**: TP=118 FP=12 TN=461 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 166 | 27 | 4 |
| normal | 28 | 240 | 8 |
| sever | 0 | 2 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8557    0.8426    0.8491       197
      normal     0.8922    0.8696    0.8807       276
       sever     0.9077    0.9833    0.9440       120

    accuracy                         0.8836       593
   macro avg     0.8852    0.8985    0.8913       593
weighted avg     0.8832    0.8836    0.8830       593

```