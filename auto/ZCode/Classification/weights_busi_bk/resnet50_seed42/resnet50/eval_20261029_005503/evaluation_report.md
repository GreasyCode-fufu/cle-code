# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/resnet50_seed42/resnet50/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8104 |
| Accuracy | 0.8362 |
| Macro-AUROC (OvR) | 0.9463 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8348 |
| Balanced Acc | 0.7999 |
| Macro-Sensitivity | 0.7999 |
| Macro-Specificity | 0.8990 |
| Macro-Precision | 0.8258 |
| AUC-PR (Macro) | 0.9013 |
| Cohen's Kappa | 0.7138 |
| MCC | 0.7163 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 1453.4 (0.69 ms/张) |
| End-to-End FPS | 172.2 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9077 | 0.7843 | 0.8429 | 0.8741 | 0.8696 | 0.9460 | 0.9594 | 65 |
| malignant | 0.7419 | 0.9647 | 0.8846 | 0.8070 | 0.9111 | 0.9321 | 0.8875 | 31 |
| normal | 0.7500 | 0.9479 | 0.7500 | 0.7500 | 0.9479 | 0.9609 | 0.8571 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=11 TN=40 FN=6
- **malignant**: TP=23 FP=3 TN=82 FN=8
- **normal**: TP=15 FP=5 TN=91 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 3 | 3 |
| malignant | 6 | 23 | 2 |
| normal | 5 | 0 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8429    0.9077    0.8741        65
   malignant     0.8846    0.7419    0.8070        31
      normal     0.7500    0.7500    0.7500        20

    accuracy                         0.8362       116
   macro avg     0.8258    0.7999    0.8104       116
weighted avg     0.8380    0.8362    0.8348       116

```