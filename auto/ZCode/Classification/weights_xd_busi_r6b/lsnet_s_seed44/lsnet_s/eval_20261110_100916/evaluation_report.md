# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/lsnet_s_seed44/lsnet_s/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7290 |
| Accuracy | 0.7627 |
| Macro-AUROC (OvR) | 0.9114 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7655 |
| Balanced Acc | 0.7406 |
| Macro-Sensitivity | 0.7406 |
| Macro-Specificity | 0.8729 |
| Macro-Precision | 0.7202 |
| AUC-PR (Macro) | 0.8359 |
| Cohen's Kappa | 0.6027 |
| MCC | 0.6037 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 1554.7 (0.64 ms/张) |
| End-to-End FPS | 175.8 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8030 | 0.8269 | 0.8548 | 0.8281 | 0.7679 | 0.8944 | 0.9111 | 66 |
| malignant | 0.7188 | 0.8837 | 0.6970 | 0.7077 | 0.8941 | 0.9086 | 0.8153 | 32 |
| normal | 0.7000 | 0.9082 | 0.6087 | 0.6512 | 0.9368 | 0.9311 | 0.7815 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=53 FP=9 TN=43 FN=13
- **malignant**: TP=23 FP=10 TN=76 FN=9
- **normal**: TP=14 FP=9 TN=89 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 53 | 7 | 6 |
| malignant | 6 | 23 | 3 |
| normal | 3 | 3 | 14 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8548    0.8030    0.8281        66
   malignant     0.6970    0.7188    0.7077        32
      normal     0.6087    0.7000    0.6512        20

    accuracy                         0.7627       118
   macro avg     0.7202    0.7406    0.7290       118
weighted avg     0.7703    0.7627    0.7655       118

```