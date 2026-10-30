# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/resnet50_seed43/resnet50/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8385 |
| Accuracy | 0.8621 |
| Macro-AUROC (OvR) | 0.9406 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8614 |
| Balanced Acc | 0.8321 |
| Macro-Sensitivity | 0.8321 |
| Macro-Specificity | 0.9181 |
| Macro-Precision | 0.8456 |
| AUC-PR (Macro) | 0.8777 |
| Cohen's Kappa | 0.7618 |
| MCC | 0.7621 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 1452.0 (0.69 ms/张) |
| End-to-End FPS | 172.1 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9077 | 0.8431 | 0.8806 | 0.8939 | 0.8776 | 0.9460 | 0.9554 | 65 |
| malignant | 0.8387 | 0.9529 | 0.8667 | 0.8525 | 0.9419 | 0.9378 | 0.8532 | 31 |
| normal | 0.7500 | 0.9583 | 0.7895 | 0.7692 | 0.9485 | 0.9380 | 0.8244 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=8 TN=43 FN=6
- **malignant**: TP=26 FP=4 TN=81 FN=5
- **normal**: TP=15 FP=4 TN=92 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 3 | 3 |
| malignant | 4 | 26 | 1 |
| normal | 4 | 1 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8806    0.9077    0.8939        65
   malignant     0.8667    0.8387    0.8525        31
      normal     0.7895    0.7500    0.7692        20

    accuracy                         0.8621       116
   macro avg     0.8456    0.8321    0.8385       116
weighted avg     0.8612    0.8621    0.8614       116

```