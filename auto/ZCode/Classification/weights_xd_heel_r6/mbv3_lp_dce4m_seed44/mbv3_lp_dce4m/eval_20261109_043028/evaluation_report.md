# mbv3_lp_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_heel_r6/mbv3_lp_dce4m_seed44/mbv3_lp_dce4m/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7962 |
| Accuracy | 0.7757 |
| Macro-AUROC (OvR) | 0.9086 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7765 |
| Balanced Acc | 0.7979 |
| Macro-Sensitivity | 0.7979 |
| Macro-Specificity | 0.8777 |
| Macro-Precision | 0.7958 |
| AUC-PR (Macro) | 0.8539 |
| Cohen's Kappa | 0.6470 |
| MCC | 0.6478 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 2425.8 (0.41 ms/张) |
| End-to-End FPS | 15.9 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.7462 | 0.8308 | 0.6869 | 0.7153 | 0.8681 | 0.8756 | 0.7766 | 197 |
| normal | 0.7391 | 0.8233 | 0.7846 | 0.7612 | 0.7838 | 0.8598 | 0.8402 | 276 |
| sever | 0.9083 | 0.9789 | 0.9160 | 0.9121 | 0.9768 | 0.9904 | 0.9447 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=147 FP=67 TN=329 FN=50
- **normal**: TP=204 FP=56 TN=261 FN=72
- **sever**: TP=109 FP=10 TN=463 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 147 | 50 | 0 |
| normal | 62 | 204 | 10 |
| sever | 5 | 6 | 109 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.6869    0.7462    0.7153       197
      normal     0.7846    0.7391    0.7612       276
       sever     0.9160    0.9083    0.9121       120

    accuracy                         0.7757       593
   macro avg     0.7958    0.7979    0.7962       593
weighted avg     0.7787    0.7757    0.7765       593

```