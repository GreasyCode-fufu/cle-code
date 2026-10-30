# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/swin_tiny_vanilla_patch4_window7_224_seed43/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8395 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9581 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8561 |
| Balanced Acc | 0.8596 |
| Macro-Sensitivity | 0.8596 |
| Macro-Specificity | 0.9222 |
| Macro-Precision | 0.8314 |
| AUC-PR (Macro) | 0.9268 |
| Cohen's Kappa | 0.7578 |
| MCC | 0.7605 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1318.4 (0.76 ms/张) |
| End-to-End FPS | 177.6 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8846 | 0.9062 | 0.8923 | 0.8519 | 0.9529 | 0.9654 | 66 |
| malignant | 0.7500 | 0.9535 | 0.8571 | 0.8000 | 0.9111 | 0.9622 | 0.9151 | 32 |
| normal | 0.9500 | 0.9286 | 0.7308 | 0.8261 | 0.9891 | 0.9592 | 0.8999 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=6 TN=46 FN=8
- **malignant**: TP=24 FP=4 TN=82 FN=8
- **normal**: TP=19 FP=7 TN=91 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 4 | 4 |
| malignant | 5 | 24 | 3 |
| normal | 1 | 0 | 19 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9062    0.8788    0.8923        66
   malignant     0.8571    0.7500    0.8000        32
      normal     0.7308    0.9500    0.8261        20

    accuracy                         0.8559       118
   macro avg     0.8314    0.8596    0.8395       118
weighted avg     0.8632    0.8559    0.8561       118

```