# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz8_dce4m_seed43/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8476 |
| Accuracy | 0.8644 |
| Macro-AUROC (OvR) | 0.9488 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8633 |
| Balanced Acc | 0.8289 |
| Macro-Sensitivity | 0.8289 |
| Macro-Specificity | 0.9157 |
| Macro-Precision | 0.8738 |
| AUC-PR (Macro) | 0.9156 |
| Cohen's Kappa | 0.7628 |
| MCC | 0.7646 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1370.8 (0.73 ms/张) |
| End-to-End FPS | 12.6 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9242 | 0.8269 | 0.8714 | 0.8971 | 0.8958 | 0.9423 | 0.9542 | 66 |
| malignant | 0.8125 | 0.9302 | 0.8125 | 0.8125 | 0.9302 | 0.9517 | 0.8925 | 32 |
| normal | 0.7500 | 0.9898 | 0.9375 | 0.8333 | 0.9510 | 0.9526 | 0.9000 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=9 TN=43 FN=5
- **malignant**: TP=26 FP=6 TN=80 FN=6
- **normal**: TP=15 FP=1 TN=97 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 5 | 0 |
| malignant | 5 | 26 | 1 |
| normal | 4 | 1 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8714    0.9242    0.8971        66
   malignant     0.8125    0.8125    0.8125        32
      normal     0.9375    0.7500    0.8333        20

    accuracy                         0.8644       118
   macro avg     0.8738    0.8289    0.8476       118
weighted avg     0.8666    0.8644    0.8633       118

```