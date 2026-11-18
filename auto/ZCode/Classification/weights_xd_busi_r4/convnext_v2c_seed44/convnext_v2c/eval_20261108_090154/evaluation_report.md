# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r4/convnext_v2c_seed44/convnext_v2c/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8366 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9520 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8589 |
| Balanced Acc | 0.8596 |
| Macro-Sensitivity | 0.8596 |
| Macro-Specificity | 0.9236 |
| Macro-Precision | 0.8405 |
| AUC-PR (Macro) | 0.9218 |
| Cohen's Kappa | 0.7589 |
| MCC | 0.7651 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 740.5 (1.35 ms/张) |
| End-to-End FPS | 148.0 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8846 | 0.9062 | 0.8923 | 0.8519 | 0.9388 | 0.9568 | 66 |
| malignant | 0.7500 | 0.9884 | 0.9600 | 0.8421 | 0.9140 | 0.9433 | 0.8841 | 32 |
| normal | 0.9500 | 0.8980 | 0.6552 | 0.7755 | 0.9888 | 0.9740 | 0.9247 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=6 TN=46 FN=8
- **malignant**: TP=24 FP=1 TN=85 FN=8
- **normal**: TP=19 FP=10 TN=88 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 1 | 7 |
| malignant | 5 | 24 | 3 |
| normal | 1 | 0 | 19 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9062    0.8788    0.8923        66
   malignant     0.9600    0.7500    0.8421        32
      normal     0.6552    0.9500    0.7755        20

    accuracy                         0.8559       118
   macro avg     0.8405    0.8596    0.8366       118
weighted avg     0.8783    0.8559    0.8589       118

```