# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/fast_t0_seed42/fast_t0/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7856 |
| Accuracy | 0.8051 |
| Macro-AUROC (OvR) | 0.9253 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8037 |
| Balanced Acc | 0.7712 |
| Macro-Sensitivity | 0.7712 |
| Macro-Specificity | 0.8793 |
| Macro-Precision | 0.8040 |
| AUC-PR (Macro) | 0.8701 |
| Cohen's Kappa | 0.6596 |
| MCC | 0.6608 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 2815.5 (0.36 ms/张) |
| End-to-End FPS | 191.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8636 | 0.7500 | 0.8143 | 0.8382 | 0.8125 | 0.8995 | 0.9325 | 66 |
| malignant | 0.7500 | 0.9186 | 0.7742 | 0.7619 | 0.9080 | 0.9273 | 0.8345 | 32 |
| normal | 0.7000 | 0.9694 | 0.8235 | 0.7568 | 0.9406 | 0.9490 | 0.8433 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=13 TN=39 FN=9
- **malignant**: TP=24 FP=7 TN=79 FN=8
- **normal**: TP=14 FP=3 TN=95 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 7 | 2 |
| malignant | 7 | 24 | 1 |
| normal | 6 | 0 | 14 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8143    0.8636    0.8382        66
   malignant     0.7742    0.7500    0.7619        32
      normal     0.8235    0.7000    0.7568        20

    accuracy                         0.8051       118
   macro avg     0.8040    0.7712    0.7856       118
weighted avg     0.8050    0.8051    0.8037       118

```