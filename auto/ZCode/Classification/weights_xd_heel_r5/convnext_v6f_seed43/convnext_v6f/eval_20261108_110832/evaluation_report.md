# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r5/convnext_v6f_seed43/convnext_v6f/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9002 |
| Accuracy | 0.8887 |
| Macro-AUROC (OvR) | 0.9809 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8895 |
| Balanced Acc | 0.8956 |
| Macro-Sensitivity | 0.8956 |
| Macro-Specificity | 0.9379 |
| Macro-Precision | 0.9059 |
| AUC-PR (Macro) | 0.9731 |
| Cohen's Kappa | 0.8235 |
| MCC | 0.8239 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 1435.1 (0.70 ms/张) |
| End-to-End FPS | 315.7 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8731 | 0.9116 | 0.8309 | 0.8515 | 0.9352 | 0.9717 | 0.9539 | 197 |
| normal | 0.8804 | 0.9022 | 0.8869 | 0.8836 | 0.8966 | 0.9713 | 0.9672 | 276 |
| sever | 0.9333 | 1.0000 | 1.0000 | 0.9655 | 0.9834 | 0.9996 | 0.9983 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=172 FP=35 TN=361 FN=25
- **normal**: TP=243 FP=31 TN=286 FN=33
- **sever**: TP=112 FP=0 TN=473 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 172 | 25 | 0 |
| normal | 33 | 243 | 0 |
| sever | 2 | 6 | 112 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8309    0.8731    0.8515       197
      normal     0.8869    0.8804    0.8836       276
       sever     1.0000    0.9333    0.9655       120

    accuracy                         0.8887       593
   macro avg     0.9059    0.8956    0.9002       593
weighted avg     0.8912    0.8887    0.8895       593

```