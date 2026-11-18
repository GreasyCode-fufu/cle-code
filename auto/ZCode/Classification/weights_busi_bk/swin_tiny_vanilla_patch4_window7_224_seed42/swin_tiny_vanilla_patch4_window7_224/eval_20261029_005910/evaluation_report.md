# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/swin_tiny_vanilla_patch4_window7_224_seed42/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8671 |
| Accuracy | 0.8966 |
| Macro-AUROC (OvR) | 0.9636 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8965 |
| Balanced Acc | 0.8642 |
| Macro-Sensitivity | 0.8642 |
| Macro-Specificity | 0.9421 |
| Macro-Precision | 0.8719 |
| AUC-PR (Macro) | 0.9336 |
| Cohen's Kappa | 0.8218 |
| MCC | 0.8226 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1316.3 (0.76 ms/张) |
| End-to-End FPS | 168.3 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.9020 | 0.9254 | 0.9394 | 0.9388 | 0.9649 | 0.9517 | 65 |
| malignant | 0.8387 | 0.9765 | 0.9286 | 0.8814 | 0.9432 | 0.9776 | 0.9519 | 31 |
| normal | 0.8000 | 0.9479 | 0.7619 | 0.7805 | 0.9579 | 0.9484 | 0.8972 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=5 TN=46 FN=3
- **malignant**: TP=26 FP=2 TN=83 FN=5
- **normal**: TP=16 FP=5 TN=91 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 2 | 1 |
| malignant | 1 | 26 | 4 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9254    0.9538    0.9394        65
   malignant     0.9286    0.8387    0.8814        31
      normal     0.7619    0.8000    0.7805        20

    accuracy                         0.8966       116
   macro avg     0.8719    0.8642    0.8671       116
weighted avg     0.8980    0.8966    0.8965       116

```