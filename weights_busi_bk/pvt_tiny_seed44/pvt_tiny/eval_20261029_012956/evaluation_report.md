# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/pvt_tiny_seed44/pvt_tiny/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7984 |
| Accuracy | 0.8220 |
| Macro-AUROC (OvR) | 0.9466 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8237 |
| Balanced Acc | 0.8224 |
| Macro-Sensitivity | 0.8224 |
| Macro-Specificity | 0.9077 |
| Macro-Precision | 0.7877 |
| AUC-PR (Macro) | 0.8811 |
| Cohen's Kappa | 0.7037 |
| MCC | 0.7070 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2078.4 (0.48 ms/张) |
| End-to-End FPS | 186.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.8846 | 0.9032 | 0.8750 | 0.8214 | 0.9344 | 0.9568 | 66 |
| malignant | 0.7188 | 0.9302 | 0.7931 | 0.7541 | 0.8989 | 0.9499 | 0.8429 | 32 |
| normal | 0.9000 | 0.9082 | 0.6667 | 0.7660 | 0.9780 | 0.9556 | 0.8436 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=6 TN=46 FN=10
- **malignant**: TP=23 FP=6 TN=80 FN=9
- **normal**: TP=18 FP=9 TN=89 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 6 | 4 |
| malignant | 4 | 23 | 5 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9032    0.8485    0.8750        66
   malignant     0.7931    0.7188    0.7541        32
      normal     0.6667    0.9000    0.7660        20

    accuracy                         0.8220       118
   macro avg     0.7877    0.8224    0.7984       118
weighted avg     0.8333    0.8220    0.8237       118

```