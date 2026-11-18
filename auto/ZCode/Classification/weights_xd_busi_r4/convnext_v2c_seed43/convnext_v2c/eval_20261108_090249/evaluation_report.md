# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r4/convnext_v2c_seed43/convnext_v2c/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8287 |
| Accuracy | 0.8475 |
| Macro-AUROC (OvR) | 0.9413 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8492 |
| Balanced Acc | 0.8483 |
| Macro-Sensitivity | 0.8483 |
| Macro-Specificity | 0.9188 |
| Macro-Precision | 0.8197 |
| AUC-PR (Macro) | 0.8961 |
| Cohen's Kappa | 0.7447 |
| MCC | 0.7472 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1289.4 (0.78 ms/张) |
| End-to-End FPS | 176.9 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8636 | 0.8846 | 0.9048 | 0.8837 | 0.8364 | 0.9235 | 0.9493 | 66 |
| malignant | 0.7812 | 0.9535 | 0.8621 | 0.8197 | 0.9213 | 0.9299 | 0.8313 | 32 |
| normal | 0.9000 | 0.9184 | 0.6923 | 0.7826 | 0.9783 | 0.9704 | 0.9078 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=6 TN=46 FN=9
- **malignant**: TP=25 FP=4 TN=82 FN=7
- **normal**: TP=18 FP=8 TN=90 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 4 | 5 |
| malignant | 4 | 25 | 3 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9048    0.8636    0.8837        66
   malignant     0.8621    0.7812    0.8197        32
      normal     0.6923    0.9000    0.7826        20

    accuracy                         0.8475       118
   macro avg     0.8197    0.8483    0.8287       118
weighted avg     0.8572    0.8475    0.8492       118

```