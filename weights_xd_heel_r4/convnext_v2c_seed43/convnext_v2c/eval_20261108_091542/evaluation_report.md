# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r4/convnext_v2c_seed43/convnext_v2c/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9028 |
| Accuracy | 0.8904 |
| Macro-AUROC (OvR) | 0.9803 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8904 |
| Balanced Acc | 0.9058 |
| Macro-Sensitivity | 0.9058 |
| Macro-Specificity | 0.9400 |
| Macro-Precision | 0.9002 |
| AUC-PR (Macro) | 0.9719 |
| Cohen's Kappa | 0.8273 |
| MCC | 0.8275 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 575.6 (1.74 ms/张) |
| End-to-End FPS | 239.8 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8680 | 0.9167 | 0.8382 | 0.8529 | 0.9332 | 0.9696 | 0.9495 | 197 |
| normal | 0.8659 | 0.9117 | 0.8951 | 0.8803 | 0.8865 | 0.9717 | 0.9679 | 276 |
| sever | 0.9833 | 0.9915 | 0.9672 | 0.9752 | 0.9958 | 0.9996 | 0.9984 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=171 FP=33 TN=363 FN=26
- **normal**: TP=239 FP=28 TN=289 FN=37
- **sever**: TP=118 FP=4 TN=469 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 171 | 26 | 0 |
| normal | 33 | 239 | 4 |
| sever | 0 | 2 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8382    0.8680    0.8529       197
      normal     0.8951    0.8659    0.8803       276
       sever     0.9672    0.9833    0.9752       120

    accuracy                         0.8904       593
   macro avg     0.9002    0.9058    0.9028       593
weighted avg     0.8908    0.8904    0.8904       593

```