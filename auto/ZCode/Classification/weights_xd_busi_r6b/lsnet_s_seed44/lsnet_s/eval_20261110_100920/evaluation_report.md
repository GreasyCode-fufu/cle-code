# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/lsnet_s_seed44/lsnet_s/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7705 |
| Accuracy | 0.7931 |
| Macro-AUROC (OvR) | 0.8711 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7937 |
| Balanced Acc | 0.7861 |
| Macro-Sensitivity | 0.7861 |
| Macro-Specificity | 0.8881 |
| Macro-Precision | 0.7598 |
| AUC-PR (Macro) | 0.7425 |
| Cohen's Kappa | 0.6527 |
| MCC | 0.6538 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 1553.9 (0.64 ms/张) |
| End-to-End FPS | 169.2 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8308 | 0.8431 | 0.8710 | 0.8504 | 0.7963 | 0.8691 | 0.8425 | 65 |
| malignant | 0.6774 | 0.8941 | 0.7000 | 0.6885 | 0.8837 | 0.8490 | 0.7459 | 31 |
| normal | 0.8500 | 0.9271 | 0.7083 | 0.7727 | 0.9674 | 0.8953 | 0.6391 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=54 FP=8 TN=43 FN=11
- **malignant**: TP=21 FP=9 TN=76 FN=10
- **normal**: TP=17 FP=7 TN=89 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 54 | 8 | 3 |
| malignant | 6 | 21 | 4 |
| normal | 2 | 1 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8710    0.8308    0.8504        65
   malignant     0.7000    0.6774    0.6885        31
      normal     0.7083    0.8500    0.7727        20

    accuracy                         0.7931       116
   macro avg     0.7598    0.7861    0.7705       116
weighted avg     0.7972    0.7931    0.7937       116

```