# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/convnext_tiny_seed42/convnext_tiny/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8338 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9315 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8579 |
| Balanced Acc | 0.8533 |
| Macro-Sensitivity | 0.8533 |
| Macro-Specificity | 0.9252 |
| Macro-Precision | 0.8250 |
| AUC-PR (Macro) | 0.8642 |
| Cohen's Kappa | 0.7588 |
| MCC | 0.7614 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 816.1 (1.23 ms/张) |
| End-to-End FPS | 161.0 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.9038 | 0.9206 | 0.8992 | 0.8545 | 0.9130 | 0.9403 | 66 |
| malignant | 0.7812 | 0.9535 | 0.8621 | 0.8197 | 0.9213 | 0.9288 | 0.7867 | 32 |
| normal | 0.9000 | 0.9184 | 0.6923 | 0.7826 | 0.9783 | 0.9526 | 0.8655 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=5 TN=47 FN=8
- **malignant**: TP=25 FP=4 TN=82 FN=7
- **normal**: TP=18 FP=8 TN=90 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 4 | 4 |
| malignant | 3 | 25 | 4 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9206    0.8788    0.8992        66
   malignant     0.8621    0.7812    0.8197        32
      normal     0.6923    0.9000    0.7826        20

    accuracy                         0.8559       118
   macro avg     0.8250    0.8533    0.8338       118
weighted avg     0.8661    0.8559    0.8579       118

```