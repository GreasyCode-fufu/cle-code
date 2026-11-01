# convnext_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_dsd_seed43/convnext_dsd/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8061 |
| Accuracy | 0.8305 |
| Macro-AUROC (OvR) | 0.9519 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8304 |
| Balanced Acc | 0.8158 |
| Macro-Sensitivity | 0.8158 |
| Macro-Specificity | 0.9055 |
| Macro-Precision | 0.8028 |
| AUC-PR (Macro) | 0.9100 |
| Cohen's Kappa | 0.7119 |
| MCC | 0.7133 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 430.1 (2.33 ms/张) |
| End-to-End FPS | 135.6 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8462 | 0.8788 | 0.8788 | 0.8462 | 0.9391 | 0.9574 | 66 |
| malignant | 0.7188 | 0.9419 | 0.8214 | 0.7667 | 0.9000 | 0.9499 | 0.8765 | 32 |
| normal | 0.8500 | 0.9286 | 0.7083 | 0.7727 | 0.9681 | 0.9668 | 0.8963 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=8 TN=44 FN=8
- **malignant**: TP=23 FP=5 TN=81 FN=9
- **normal**: TP=17 FP=7 TN=91 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 5 | 3 |
| malignant | 5 | 23 | 4 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8788    0.8788    0.8788        66
   malignant     0.8214    0.7188    0.7667        32
      normal     0.7083    0.8500    0.7727        20

    accuracy                         0.8305       118
   macro avg     0.8028    0.8158    0.8061       118
weighted avg     0.8343    0.8305    0.8304       118

```