# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r5/convnext_v6f_seed43/convnext_v6f/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8538 |
| Accuracy | 0.8644 |
| Macro-AUROC (OvR) | 0.9542 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8646 |
| Balanced Acc | 0.8638 |
| Macro-Sensitivity | 0.8638 |
| Macro-Specificity | 0.9226 |
| Macro-Precision | 0.8472 |
| AUC-PR (Macro) | 0.9096 |
| Cohen's Kappa | 0.7701 |
| MCC | 0.7708 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 908.8 (1.10 ms/张) |
| End-to-End FPS | 162.3 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8654 | 0.8923 | 0.8855 | 0.8491 | 0.9324 | 0.9586 | 66 |
| malignant | 0.8125 | 0.9535 | 0.8667 | 0.8387 | 0.9318 | 0.9502 | 0.8580 | 32 |
| normal | 0.9000 | 0.9490 | 0.7826 | 0.8372 | 0.9789 | 0.9801 | 0.9122 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=7 TN=45 FN=8
- **malignant**: TP=26 FP=4 TN=82 FN=6
- **normal**: TP=18 FP=5 TN=93 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 4 | 4 |
| malignant | 5 | 26 | 1 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8923    0.8788    0.8855        66
   malignant     0.8667    0.8125    0.8387        32
      normal     0.7826    0.9000    0.8372        20

    accuracy                         0.8644       118
   macro avg     0.8472    0.8638    0.8538       118
weighted avg     0.8668    0.8644    0.8646       118

```