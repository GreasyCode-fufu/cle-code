# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16bn_dce4m_seed42/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8030 |
| Accuracy | 0.8276 |
| Macro-AUROC (OvR) | 0.9272 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8293 |
| Balanced Acc | 0.8119 |
| Macro-Sensitivity | 0.8119 |
| Macro-Specificity | 0.9042 |
| Macro-Precision | 0.7996 |
| AUC-PR (Macro) | 0.8718 |
| Cohen's Kappa | 0.7081 |
| MCC | 0.7093 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1113.3 (0.90 ms/张) |
| End-to-End FPS | 7.1 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8615 | 0.8431 | 0.8750 | 0.8682 | 0.8269 | 0.9198 | 0.9321 | 65 |
| malignant | 0.7742 | 0.9529 | 0.8571 | 0.8136 | 0.9205 | 0.9431 | 0.8750 | 31 |
| normal | 0.8000 | 0.9167 | 0.6667 | 0.7273 | 0.9565 | 0.9187 | 0.8083 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=8 TN=43 FN=9
- **malignant**: TP=24 FP=4 TN=81 FN=7
- **normal**: TP=16 FP=8 TN=88 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 4 | 5 |
| malignant | 4 | 24 | 3 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8750    0.8615    0.8682        65
   malignant     0.8571    0.7742    0.8136        31
      normal     0.6667    0.8000    0.7273        20

    accuracy                         0.8276       116
   macro avg     0.7996    0.8119    0.8030       116
weighted avg     0.8343    0.8276    0.8293       116

```