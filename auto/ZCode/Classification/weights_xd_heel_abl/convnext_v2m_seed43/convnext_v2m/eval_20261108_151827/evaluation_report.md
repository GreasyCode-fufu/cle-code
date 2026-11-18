# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_abl/convnext_v2m_seed43/convnext_v2m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9041 |
| Accuracy | 0.8938 |
| Macro-AUROC (OvR) | 0.9796 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8938 |
| Balanced Acc | 0.9066 |
| Macro-Sensitivity | 0.9066 |
| Macro-Specificity | 0.9420 |
| Macro-Precision | 0.9017 |
| AUC-PR (Macro) | 0.9707 |
| Cohen's Kappa | 0.8324 |
| MCC | 0.8326 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 578.9 (1.73 ms/张) |
| End-to-End FPS | 238.9 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8680 | 0.9217 | 0.8465 | 0.8571 | 0.9335 | 0.9698 | 0.9518 | 197 |
| normal | 0.8768 | 0.9148 | 0.8996 | 0.8881 | 0.8951 | 0.9707 | 0.9650 | 276 |
| sever | 0.9750 | 0.9894 | 0.9590 | 0.9669 | 0.9936 | 0.9983 | 0.9953 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=171 FP=31 TN=365 FN=26
- **normal**: TP=242 FP=27 TN=290 FN=34
- **sever**: TP=117 FP=5 TN=468 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 171 | 25 | 1 |
| normal | 30 | 242 | 4 |
| sever | 1 | 2 | 117 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8465    0.8680    0.8571       197
      normal     0.8996    0.8768    0.8881       276
       sever     0.9590    0.9750    0.9669       120

    accuracy                         0.8938       593
   macro avg     0.9017    0.9066    0.9041       593
weighted avg     0.8940    0.8938    0.8938       593

```