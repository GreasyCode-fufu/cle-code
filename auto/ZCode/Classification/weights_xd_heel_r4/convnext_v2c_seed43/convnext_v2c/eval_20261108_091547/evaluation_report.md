# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r4/convnext_v2c_seed43/convnext_v2c/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8995 |
| Accuracy | 0.8904 |
| Macro-AUROC (OvR) | 0.9807 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8899 |
| Balanced Acc | 0.9104 |
| Macro-Sensitivity | 0.9104 |
| Macro-Specificity | 0.9426 |
| Macro-Precision | 0.8913 |
| AUC-PR (Macro) | 0.9716 |
| Cohen's Kappa | 0.8286 |
| MCC | 0.8300 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 816.4 (1.22 ms/张) |
| End-to-End FPS | 263.2 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8832 | 0.9141 | 0.8365 | 0.8593 | 0.9403 | 0.9716 | 0.9520 | 197 |
| normal | 0.8478 | 0.9369 | 0.9213 | 0.8830 | 0.8761 | 0.9714 | 0.9664 | 276 |
| sever | 1.0000 | 0.9767 | 0.9160 | 0.9562 | 1.0000 | 0.9992 | 0.9965 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=174 FP=34 TN=362 FN=23
- **normal**: TP=234 FP=20 TN=297 FN=42
- **sever**: TP=120 FP=11 TN=462 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 174 | 20 | 3 |
| normal | 34 | 234 | 8 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8365    0.8832    0.8593       197
      normal     0.9213    0.8478    0.8830       276
       sever     0.9160    1.0000    0.9562       120

    accuracy                         0.8904       593
   macro avg     0.8913    0.9104    0.8995       593
weighted avg     0.8921    0.8904    0.8899       593

```