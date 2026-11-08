# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz16bn_seed44/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8051 |
| Accuracy | 0.8305 |
| Macro-AUROC (OvR) | 0.9434 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8359 |
| Balanced Acc | 0.8328 |
| Macro-Sensitivity | 0.8328 |
| Macro-Specificity | 0.9155 |
| Macro-Precision | 0.8023 |
| AUC-PR (Macro) | 0.8966 |
| Cohen's Kappa | 0.7202 |
| MCC | 0.7269 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1503.5 (0.67 ms/张) |
| End-to-End FPS | 60.5 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.9038 | 0.9180 | 0.8819 | 0.8246 | 0.9347 | 0.9542 | 66 |
| malignant | 0.7500 | 0.9651 | 0.8889 | 0.8136 | 0.9121 | 0.9368 | 0.8688 | 32 |
| normal | 0.9000 | 0.8776 | 0.6000 | 0.7200 | 0.9773 | 0.9587 | 0.8668 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=5 TN=47 FN=10
- **malignant**: TP=24 FP=3 TN=83 FN=8
- **normal**: TP=18 FP=12 TN=86 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 3 | 7 |
| malignant | 3 | 24 | 5 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9180    0.8485    0.8819        66
   malignant     0.8889    0.7500    0.8136        32
      normal     0.6000    0.9000    0.7200        20

    accuracy                         0.8305       118
   macro avg     0.8023    0.8328    0.8051       118
weighted avg     0.8562    0.8305    0.8359       118

```