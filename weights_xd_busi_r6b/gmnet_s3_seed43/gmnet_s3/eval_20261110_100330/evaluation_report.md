# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/gmnet_s3_seed43/gmnet_s3/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8135 |
| Accuracy | 0.8390 |
| Macro-AUROC (OvR) | 0.9322 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8422 |
| Balanced Acc | 0.8379 |
| Macro-Sensitivity | 0.8379 |
| Macro-Specificity | 0.9184 |
| Macro-Precision | 0.8065 |
| AUC-PR (Macro) | 0.8469 |
| Cohen's Kappa | 0.7323 |
| MCC | 0.7367 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1365.3 (0.73 ms/张) |
| End-to-End FPS | 170.5 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8636 | 0.9038 | 0.9194 | 0.8906 | 0.8393 | 0.9291 | 0.9563 | 66 |
| malignant | 0.7500 | 0.9535 | 0.8571 | 0.8000 | 0.9111 | 0.9144 | 0.8661 | 32 |
| normal | 0.9000 | 0.8980 | 0.6429 | 0.7500 | 0.9778 | 0.9531 | 0.7182 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=5 TN=47 FN=9
- **malignant**: TP=24 FP=4 TN=82 FN=8
- **normal**: TP=18 FP=10 TN=88 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 4 | 5 |
| malignant | 3 | 24 | 5 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9194    0.8636    0.8906        66
   malignant     0.8571    0.7500    0.8000        32
      normal     0.6429    0.9000    0.7500        20

    accuracy                         0.8390       118
   macro avg     0.8065    0.8379    0.8135       118
weighted avg     0.8556    0.8390    0.8422       118

```