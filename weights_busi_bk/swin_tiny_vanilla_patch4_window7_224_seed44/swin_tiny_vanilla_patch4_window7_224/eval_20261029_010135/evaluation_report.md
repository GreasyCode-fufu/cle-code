# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/swin_tiny_vanilla_patch4_window7_224_seed44/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8434 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9521 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8559 |
| Balanced Acc | 0.8408 |
| Macro-Sensitivity | 0.8408 |
| Macro-Specificity | 0.9153 |
| Macro-Precision | 0.8464 |
| AUC-PR (Macro) | 0.9182 |
| Cohen's Kappa | 0.7533 |
| MCC | 0.7534 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1320.1 (0.76 ms/张) |
| End-to-End FPS | 177.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8462 | 0.8788 | 0.8788 | 0.8462 | 0.9311 | 0.9541 | 66 |
| malignant | 0.8438 | 0.9302 | 0.8182 | 0.8308 | 0.9412 | 0.9589 | 0.9015 | 32 |
| normal | 0.8000 | 0.9694 | 0.8421 | 0.8205 | 0.9596 | 0.9663 | 0.8989 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=8 TN=44 FN=8
- **malignant**: TP=27 FP=6 TN=80 FN=5
- **normal**: TP=16 FP=3 TN=95 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 6 | 2 |
| malignant | 4 | 27 | 1 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8788    0.8788    0.8788        66
   malignant     0.8182    0.8438    0.8308        32
      normal     0.8421    0.8000    0.8205        20

    accuracy                         0.8559       118
   macro avg     0.8464    0.8408    0.8434       118
weighted avg     0.8561    0.8559    0.8559       118

```