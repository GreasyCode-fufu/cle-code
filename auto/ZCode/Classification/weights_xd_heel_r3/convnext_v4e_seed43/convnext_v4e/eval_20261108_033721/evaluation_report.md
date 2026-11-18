# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r3/convnext_v4e_seed43/convnext_v4e/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8948 |
| Accuracy | 0.8938 |
| Macro-AUROC (OvR) | 0.9803 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8927 |
| Balanced Acc | 0.9030 |
| Macro-Sensitivity | 0.9030 |
| Macro-Specificity | 0.9425 |
| Macro-Precision | 0.8954 |
| AUC-PR (Macro) | 0.9717 |
| Cohen's Kappa | 0.8323 |
| MCC | 0.8354 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 418.0 (2.39 ms/张) |
| End-to-End FPS | 205.1 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7970 | 0.9823 | 0.9573 | 0.8698 | 0.9068 | 0.9699 | 0.9527 | 197 |
| normal | 0.9203 | 0.8896 | 0.8789 | 0.8991 | 0.9276 | 0.9726 | 0.9681 | 276 |
| sever | 0.9917 | 0.9556 | 0.8500 | 0.9154 | 0.9978 | 0.9984 | 0.9944 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=157 FP=7 TN=389 FN=40
- **normal**: TP=254 FP=35 TN=282 FN=22
- **sever**: TP=119 FP=21 TN=452 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 157 | 34 | 6 |
| normal | 7 | 254 | 15 |
| sever | 0 | 1 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.9573    0.7970    0.8698       197
      normal     0.8789    0.9203    0.8991       276
       sever     0.8500    0.9917    0.9154       120

    accuracy                         0.8938       593
   macro avg     0.8954    0.9030    0.8948       593
weighted avg     0.8991    0.8938    0.8927       593

```