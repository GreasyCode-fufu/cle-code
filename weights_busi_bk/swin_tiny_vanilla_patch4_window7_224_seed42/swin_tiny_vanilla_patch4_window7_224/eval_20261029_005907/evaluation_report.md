# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/swin_tiny_vanilla_patch4_window7_224_seed42/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8503 |
| Accuracy | 0.8644 |
| Macro-AUROC (OvR) | 0.9411 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8642 |
| Balanced Acc | 0.8584 |
| Macro-Sensitivity | 0.8584 |
| Macro-Specificity | 0.9226 |
| Macro-Precision | 0.8462 |
| AUC-PR (Macro) | 0.8849 |
| Cohen's Kappa | 0.7692 |
| MCC | 0.7700 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1323.8 (0.76 ms/张) |
| End-to-End FPS | 177.4 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.8654 | 0.8939 | 0.8939 | 0.8654 | 0.9295 | 0.9479 | 66 |
| malignant | 0.7812 | 0.9535 | 0.8621 | 0.8197 | 0.9213 | 0.9197 | 0.8053 | 32 |
| normal | 0.9000 | 0.9490 | 0.7826 | 0.8372 | 0.9789 | 0.9740 | 0.9014 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=7 TN=45 FN=7
- **malignant**: TP=25 FP=4 TN=82 FN=7
- **normal**: TP=18 FP=5 TN=93 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 4 | 3 |
| malignant | 5 | 25 | 2 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8939    0.8939    0.8939        66
   malignant     0.8621    0.7812    0.8197        32
      normal     0.7826    0.9000    0.8372        20

    accuracy                         0.8644       118
   macro avg     0.8462    0.8584    0.8503       118
weighted avg     0.8664    0.8644    0.8642       118

```