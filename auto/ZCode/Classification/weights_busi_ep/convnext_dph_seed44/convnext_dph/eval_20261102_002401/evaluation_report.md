# convnext_dph 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_dph_seed44/convnext_dph/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8137 |
| Accuracy | 0.8305 |
| Macro-AUROC (OvR) | 0.9461 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8305 |
| Balanced Acc | 0.8212 |
| Macro-Sensitivity | 0.8212 |
| Macro-Specificity | 0.9025 |
| Macro-Precision | 0.8101 |
| AUC-PR (Macro) | 0.9003 |
| Cohen's Kappa | 0.7115 |
| MCC | 0.7123 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 406.8 (2.46 ms/张) |
| End-to-End FPS | 133.1 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8636 | 0.8269 | 0.8636 | 0.8636 | 0.8269 | 0.9213 | 0.9468 | 66 |
| malignant | 0.7500 | 0.9419 | 0.8276 | 0.7869 | 0.9101 | 0.9528 | 0.8682 | 32 |
| normal | 0.8500 | 0.9388 | 0.7391 | 0.7907 | 0.9684 | 0.9643 | 0.8858 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=9 TN=43 FN=9
- **malignant**: TP=24 FP=5 TN=81 FN=8
- **normal**: TP=17 FP=6 TN=92 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 5 | 4 |
| malignant | 6 | 24 | 2 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8636    0.8636    0.8636        66
   malignant     0.8276    0.7500    0.7869        32
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.8305       118
   macro avg     0.8101    0.8212    0.8137       118
weighted avg     0.8328    0.8305    0.8305       118

```