# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/ghostnet_seed44/ghostnet/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7429 |
| Accuracy | 0.7712 |
| Macro-AUROC (OvR) | 0.9369 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7737 |
| Balanced Acc | 0.7573 |
| Macro-Sensitivity | 0.7573 |
| Macro-Specificity | 0.8768 |
| Macro-Precision | 0.7329 |
| AUC-PR (Macro) | 0.8788 |
| Cohen's Kappa | 0.6174 |
| MCC | 0.6188 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 1383.1 (0.72 ms/张) |
| End-to-End FPS | 178.2 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8030 | 0.8269 | 0.8548 | 0.8281 | 0.7679 | 0.9155 | 0.9422 | 66 |
| malignant | 0.7188 | 0.8953 | 0.7188 | 0.7188 | 0.8953 | 0.9259 | 0.8020 | 32 |
| normal | 0.7500 | 0.9082 | 0.6250 | 0.6818 | 0.9468 | 0.9694 | 0.8922 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=53 FP=9 TN=43 FN=13
- **malignant**: TP=23 FP=9 TN=77 FN=9
- **normal**: TP=15 FP=9 TN=89 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 53 | 8 | 5 |
| malignant | 5 | 23 | 4 |
| normal | 4 | 1 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8548    0.8030    0.8281        66
   malignant     0.7188    0.7188    0.7188        32
      normal     0.6250    0.7500    0.6818        20

    accuracy                         0.7712       118
   macro avg     0.7329    0.7573    0.7429       118
weighted avg     0.7790    0.7712    0.7737       118

```