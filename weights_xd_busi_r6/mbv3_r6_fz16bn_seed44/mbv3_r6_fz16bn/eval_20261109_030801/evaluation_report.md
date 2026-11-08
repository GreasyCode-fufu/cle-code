# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz16bn_seed44/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8093 |
| Accuracy | 0.8362 |
| Macro-AUROC (OvR) | 0.9464 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8427 |
| Balanced Acc | 0.8401 |
| Macro-Sensitivity | 0.8401 |
| Macro-Specificity | 0.9230 |
| Macro-Precision | 0.8018 |
| AUC-PR (Macro) | 0.9092 |
| Cohen's Kappa | 0.7318 |
| MCC | 0.7393 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 997.3 (1.00 ms/张) |
| End-to-End FPS | 52.2 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8462 | 0.9412 | 0.9483 | 0.8943 | 0.8276 | 0.9611 | 0.9673 | 65 |
| malignant | 0.7742 | 0.9529 | 0.8571 | 0.8136 | 0.9205 | 0.9249 | 0.8781 | 31 |
| normal | 0.9000 | 0.8750 | 0.6000 | 0.7200 | 0.9767 | 0.9531 | 0.8821 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=55 FP=3 TN=48 FN=10
- **malignant**: TP=24 FP=4 TN=81 FN=7
- **normal**: TP=18 FP=12 TN=84 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 55 | 4 | 6 |
| malignant | 1 | 24 | 6 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9483    0.8462    0.8943        65
   malignant     0.8571    0.7742    0.8136        31
      normal     0.6000    0.9000    0.7200        20

    accuracy                         0.8362       116
   macro avg     0.8018    0.8401    0.8093       116
weighted avg     0.8639    0.8362    0.8427       116

```