# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r5/convnext_v6f_seed44/convnext_v6f/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8321 |
| Accuracy | 0.8475 |
| Macro-AUROC (OvR) | 0.9517 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8495 |
| Balanced Acc | 0.8537 |
| Macro-Sensitivity | 0.8537 |
| Macro-Specificity | 0.9188 |
| Macro-Precision | 0.8207 |
| AUC-PR (Macro) | 0.9103 |
| Cohen's Kappa | 0.7457 |
| MCC | 0.7484 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 1244.0 (0.80 ms/张) |
| End-to-End FPS | 175.2 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.8846 | 0.9032 | 0.8750 | 0.8214 | 0.9266 | 0.9540 | 66 |
| malignant | 0.8125 | 0.9535 | 0.8667 | 0.8387 | 0.9318 | 0.9520 | 0.8374 | 32 |
| normal | 0.9000 | 0.9184 | 0.6923 | 0.7826 | 0.9783 | 0.9765 | 0.9395 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=6 TN=46 FN=10
- **malignant**: TP=26 FP=4 TN=82 FN=6
- **normal**: TP=18 FP=8 TN=90 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 4 | 6 |
| malignant | 4 | 26 | 2 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9032    0.8485    0.8750        66
   malignant     0.8667    0.8125    0.8387        32
      normal     0.6923    0.9000    0.7826        20

    accuracy                         0.8475       118
   macro avg     0.8207    0.8537    0.8321       118
weighted avg     0.8576    0.8475    0.8495       118

```