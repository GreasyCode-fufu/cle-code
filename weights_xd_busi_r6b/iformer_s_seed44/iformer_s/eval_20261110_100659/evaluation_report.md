# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/iformer_s_seed44/iformer_s/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7325 |
| Accuracy | 0.7627 |
| Macro-AUROC (OvR) | 0.9240 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7652 |
| Balanced Acc | 0.7468 |
| Macro-Sensitivity | 0.7468 |
| Macro-Specificity | 0.8729 |
| Macro-Precision | 0.7224 |
| AUC-PR (Macro) | 0.8222 |
| Cohen's Kappa | 0.6033 |
| MCC | 0.6046 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 3008.7 (0.33 ms/张) |
| End-to-End FPS | 181.6 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8030 | 0.8269 | 0.8548 | 0.8281 | 0.7679 | 0.9213 | 0.9500 | 66 |
| malignant | 0.6875 | 0.8837 | 0.6875 | 0.6875 | 0.8837 | 0.9128 | 0.8497 | 32 |
| normal | 0.7500 | 0.9082 | 0.6250 | 0.6818 | 0.9468 | 0.9378 | 0.6669 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=53 FP=9 TN=43 FN=13
- **malignant**: TP=22 FP=10 TN=76 FN=10
- **normal**: TP=15 FP=9 TN=89 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 53 | 8 | 5 |
| malignant | 6 | 22 | 4 |
| normal | 3 | 2 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8548    0.8030    0.8281        66
   malignant     0.6875    0.6875    0.6875        32
      normal     0.6250    0.7500    0.6818        20

    accuracy                         0.7627       118
   macro avg     0.7224    0.7468    0.7325       118
weighted avg     0.7705    0.7627    0.7652       118

```