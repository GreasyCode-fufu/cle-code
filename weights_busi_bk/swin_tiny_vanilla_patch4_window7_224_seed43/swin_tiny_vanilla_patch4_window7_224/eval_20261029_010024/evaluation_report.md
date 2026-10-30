# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/swin_tiny_vanilla_patch4_window7_224_seed43/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8595 |
| Accuracy | 0.8879 |
| Macro-AUROC (OvR) | 0.9549 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8880 |
| Balanced Acc | 0.8591 |
| Macro-Sensitivity | 0.8591 |
| Macro-Specificity | 0.9382 |
| Macro-Precision | 0.8609 |
| AUC-PR (Macro) | 0.9234 |
| Cohen's Kappa | 0.8078 |
| MCC | 0.8081 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1315.4 (0.76 ms/张) |
| End-to-End FPS | 170.4 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.9020 | 0.9242 | 0.9313 | 0.9200 | 0.9611 | 0.9380 | 65 |
| malignant | 0.8387 | 0.9647 | 0.8966 | 0.8667 | 0.9425 | 0.9780 | 0.9522 | 31 |
| normal | 0.8000 | 0.9479 | 0.7619 | 0.7805 | 0.9579 | 0.9255 | 0.8800 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=5 TN=46 FN=4
- **malignant**: TP=26 FP=3 TN=82 FN=5
- **normal**: TP=16 FP=5 TN=91 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 3 | 1 |
| malignant | 1 | 26 | 4 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9242    0.9385    0.9313        65
   malignant     0.8966    0.8387    0.8667        31
      normal     0.7619    0.8000    0.7805        20

    accuracy                         0.8879       116
   macro avg     0.8609    0.8591    0.8595       116
weighted avg     0.8889    0.8879    0.8880       116

```