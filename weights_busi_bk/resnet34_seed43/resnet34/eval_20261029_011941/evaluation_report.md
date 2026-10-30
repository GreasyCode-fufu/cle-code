# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/resnet34_seed43/resnet34/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8008 |
| Accuracy | 0.8276 |
| Macro-AUROC (OvR) | 0.9405 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8295 |
| Balanced Acc | 0.8119 |
| Macro-Sensitivity | 0.8119 |
| Macro-Specificity | 0.9069 |
| Macro-Precision | 0.7944 |
| AUC-PR (Macro) | 0.8957 |
| Cohen's Kappa | 0.7093 |
| MCC | 0.7105 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 1923.8 (0.52 ms/张) |
| End-to-End FPS | 177.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8615 | 0.8627 | 0.8889 | 0.8750 | 0.8302 | 0.9496 | 0.9586 | 65 |
| malignant | 0.7742 | 0.9412 | 0.8276 | 0.8000 | 0.9195 | 0.9328 | 0.8742 | 31 |
| normal | 0.8000 | 0.9167 | 0.6667 | 0.7273 | 0.9565 | 0.9391 | 0.8543 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=7 TN=44 FN=9
- **malignant**: TP=24 FP=5 TN=80 FN=7
- **normal**: TP=16 FP=8 TN=88 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 4 | 5 |
| malignant | 4 | 24 | 3 |
| normal | 3 | 1 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8889    0.8615    0.8750        65
   malignant     0.8276    0.7742    0.8000        31
      normal     0.6667    0.8000    0.7273        20

    accuracy                         0.8276       116
   macro avg     0.7944    0.8119    0.8008       116
weighted avg     0.8342    0.8276    0.8295       116

```