# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/gmnet_s3_seed42/gmnet_s3/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8013 |
| Accuracy | 0.8220 |
| Macro-AUROC (OvR) | 0.9528 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8233 |
| Balanced Acc | 0.8162 |
| Macro-Sensitivity | 0.8162 |
| Macro-Specificity | 0.9021 |
| Macro-Precision | 0.7942 |
| AUC-PR (Macro) | 0.9147 |
| Cohen's Kappa | 0.7004 |
| MCC | 0.7021 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1284.9 (0.78 ms/张) |
| End-to-End FPS | 170.0 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.8462 | 0.8750 | 0.8615 | 0.8148 | 0.9323 | 0.9543 | 66 |
| malignant | 0.7500 | 0.9419 | 0.8276 | 0.7869 | 0.9101 | 0.9573 | 0.9126 | 32 |
| normal | 0.8500 | 0.9184 | 0.6800 | 0.7556 | 0.9677 | 0.9689 | 0.8773 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=8 TN=44 FN=10
- **malignant**: TP=24 FP=5 TN=81 FN=8
- **normal**: TP=17 FP=8 TN=90 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 5 | 5 |
| malignant | 5 | 24 | 3 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8750    0.8485    0.8615        66
   malignant     0.8276    0.7500    0.7869        32
      normal     0.6800    0.8500    0.7556        20

    accuracy                         0.8220       118
   macro avg     0.7942    0.8162    0.8013       118
weighted avg     0.8291    0.8220    0.8233       118

```