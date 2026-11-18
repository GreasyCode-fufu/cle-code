# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/ghostnet_seed42/ghostnet/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7731 |
| Accuracy | 0.7881 |
| Macro-AUROC (OvR) | 0.9275 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7908 |
| Balanced Acc | 0.8013 |
| Macro-Sensitivity | 0.8013 |
| Macro-Specificity | 0.8896 |
| Macro-Precision | 0.7562 |
| AUC-PR (Macro) | 0.8520 |
| Cohen's Kappa | 0.6525 |
| MCC | 0.6569 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 1405.7 (0.71 ms/张) |
| End-to-End FPS | 179.3 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.7727 | 0.8654 | 0.8793 | 0.8226 | 0.7500 | 0.8878 | 0.9298 | 66 |
| malignant | 0.7812 | 0.8953 | 0.7353 | 0.7576 | 0.9167 | 0.9375 | 0.7958 | 32 |
| normal | 0.8500 | 0.9082 | 0.6538 | 0.7391 | 0.9674 | 0.9571 | 0.8303 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=51 FP=7 TN=45 FN=15
- **malignant**: TP=25 FP=9 TN=77 FN=7
- **normal**: TP=17 FP=9 TN=89 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 51 | 8 | 7 |
| malignant | 5 | 25 | 2 |
| normal | 2 | 1 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8793    0.7727    0.8226        66
   malignant     0.7353    0.7812    0.7576        32
      normal     0.6538    0.8500    0.7391        20

    accuracy                         0.7881       118
   macro avg     0.7562    0.8013    0.7731       118
weighted avg     0.8020    0.7881    0.7908       118

```