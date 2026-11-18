# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz16_seed44/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8063 |
| Accuracy | 0.8305 |
| Macro-AUROC (OvR) | 0.9452 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8338 |
| Balanced Acc | 0.8328 |
| Macro-Sensitivity | 0.8328 |
| Macro-Specificity | 0.9146 |
| Macro-Precision | 0.7962 |
| AUC-PR (Macro) | 0.9135 |
| Cohen's Kappa | 0.7194 |
| MCC | 0.7238 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1748.3 (0.57 ms/张) |
| End-to-End FPS | 183.5 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.9038 | 0.9180 | 0.8819 | 0.8246 | 0.9400 | 0.9511 | 66 |
| malignant | 0.7500 | 0.9419 | 0.8276 | 0.7869 | 0.9101 | 0.9415 | 0.8857 | 32 |
| normal | 0.9000 | 0.8980 | 0.6429 | 0.7500 | 0.9778 | 0.9541 | 0.9036 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=5 TN=47 FN=10
- **malignant**: TP=24 FP=5 TN=81 FN=8
- **normal**: TP=18 FP=10 TN=88 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 5 | 5 |
| malignant | 3 | 24 | 5 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9180    0.8485    0.8819        66
   malignant     0.8276    0.7500    0.7869        32
      normal     0.6429    0.9000    0.7500        20

    accuracy                         0.8305       118
   macro avg     0.7962    0.8328    0.8063       118
weighted avg     0.8469    0.8305    0.8338       118

```