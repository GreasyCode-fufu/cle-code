# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/gmnet_s3_seed42/gmnet_s3/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8376 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9363 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8695 |
| Balanced Acc | 0.8378 |
| Macro-Sensitivity | 0.8378 |
| Macro-Specificity | 0.9260 |
| Macro-Precision | 0.8545 |
| AUC-PR (Macro) | 0.8803 |
| Cohen's Kappa | 0.7763 |
| MCC | 0.7812 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1308.3 (0.76 ms/张) |
| End-to-End FPS | 165.2 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.8627 | 0.8986 | 0.9254 | 0.9362 | 0.9433 | 0.9373 | 65 |
| malignant | 0.7097 | 0.9882 | 0.9565 | 0.8148 | 0.9032 | 0.9192 | 0.8868 | 31 |
| normal | 0.8500 | 0.9271 | 0.7083 | 0.7727 | 0.9674 | 0.9464 | 0.8168 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=7 TN=44 FN=3
- **malignant**: TP=22 FP=1 TN=84 FN=9
- **normal**: TP=17 FP=7 TN=89 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 1 | 2 |
| malignant | 4 | 22 | 5 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8986    0.9538    0.9254        65
   malignant     0.9565    0.7097    0.8148        31
      normal     0.7083    0.8500    0.7727        20

    accuracy                         0.8707       116
   macro avg     0.8545    0.8378    0.8376       116
weighted avg     0.8812    0.8707    0.8695       116

```