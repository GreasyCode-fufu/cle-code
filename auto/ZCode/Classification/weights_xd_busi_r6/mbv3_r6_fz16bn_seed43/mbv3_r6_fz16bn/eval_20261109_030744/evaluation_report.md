# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz16bn_seed43/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8052 |
| Accuracy | 0.8362 |
| Macro-AUROC (OvR) | 0.9515 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8399 |
| Balanced Acc | 0.8173 |
| Macro-Sensitivity | 0.8173 |
| Macro-Specificity | 0.9121 |
| Macro-Precision | 0.8187 |
| AUC-PR (Macro) | 0.9131 |
| Cohen's Kappa | 0.7230 |
| MCC | 0.7289 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1278.8 (0.78 ms/张) |
| End-to-End FPS | 44.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8923 | 0.8627 | 0.8923 | 0.8923 | 0.8627 | 0.9563 | 0.9662 | 65 |
| malignant | 0.7097 | 0.9882 | 0.9565 | 0.8148 | 0.9032 | 0.9461 | 0.9205 | 31 |
| normal | 0.8500 | 0.8854 | 0.6071 | 0.7083 | 0.9659 | 0.9521 | 0.8525 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=7 TN=44 FN=7
- **malignant**: TP=22 FP=1 TN=84 FN=9
- **normal**: TP=17 FP=11 TN=85 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 1 | 6 |
| malignant | 4 | 22 | 5 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8923    0.8923    0.8923        65
   malignant     0.9565    0.7097    0.8148        31
      normal     0.6071    0.8500    0.7083        20

    accuracy                         0.8362       116
   macro avg     0.8187    0.8173    0.8052       116
weighted avg     0.8603    0.8362    0.8399       116

```