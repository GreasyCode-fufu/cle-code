# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mobilenet_v3_large_seed43/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8339 |
| Accuracy | 0.8246 |
| Macro-AUROC (OvR) | 0.9594 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8236 |
| Balanced Acc | 0.8544 |
| Macro-Sensitivity | 0.8544 |
| Macro-Specificity | 0.9104 |
| Macro-Precision | 0.8222 |
| AUC-PR (Macro) | 0.9386 |
| Cohen's Kappa | 0.7285 |
| MCC | 0.7324 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 4229.2 (0.24 ms/张) |
| End-to-End FPS | 347.5 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8071 | 0.8662 | 0.7500 | 0.7775 | 0.9003 | 0.9339 | 0.8868 | 197 |
| normal | 0.7645 | 0.9180 | 0.8903 | 0.8226 | 0.8174 | 0.9463 | 0.9365 | 276 |
| sever | 0.9917 | 0.9471 | 0.8264 | 0.9015 | 0.9978 | 0.9981 | 0.9924 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=159 FP=53 TN=343 FN=38
- **normal**: TP=211 FP=26 TN=291 FN=65
- **sever**: TP=119 FP=25 TN=448 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 159 | 26 | 12 |
| normal | 52 | 211 | 13 |
| sever | 1 | 0 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7500    0.8071    0.7775       197
      normal     0.8903    0.7645    0.8226       276
       sever     0.8264    0.9917    0.9015       120

    accuracy                         0.8246       593
   macro avg     0.8222    0.8544    0.8339       593
weighted avg     0.8308    0.8246    0.8236       593

```