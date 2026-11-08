# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16bn_dce4m_seed44/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8292 |
| Accuracy | 0.8534 |
| Macro-AUROC (OvR) | 0.9503 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8541 |
| Balanced Acc | 0.8329 |
| Macro-Sensitivity | 0.8329 |
| Macro-Specificity | 0.9177 |
| Macro-Precision | 0.8272 |
| AUC-PR (Macro) | 0.9167 |
| Cohen's Kappa | 0.7501 |
| MCC | 0.7505 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1556.8 (0.64 ms/张) |
| End-to-End FPS | 12.3 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8923 | 0.8627 | 0.8923 | 0.8923 | 0.8627 | 0.9433 | 0.9507 | 65 |
| malignant | 0.8065 | 0.9529 | 0.8621 | 0.8333 | 0.9310 | 0.9586 | 0.9097 | 31 |
| normal | 0.8000 | 0.9375 | 0.7273 | 0.7619 | 0.9574 | 0.9490 | 0.8897 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=7 TN=44 FN=7
- **malignant**: TP=25 FP=4 TN=81 FN=6
- **normal**: TP=16 FP=6 TN=90 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 4 | 3 |
| malignant | 3 | 25 | 3 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8923    0.8923    0.8923        65
   malignant     0.8621    0.8065    0.8333        31
      normal     0.7273    0.8000    0.7619        20

    accuracy                         0.8534       116
   macro avg     0.8272    0.8329    0.8292       116
weighted avg     0.8558    0.8534    0.8541       116

```