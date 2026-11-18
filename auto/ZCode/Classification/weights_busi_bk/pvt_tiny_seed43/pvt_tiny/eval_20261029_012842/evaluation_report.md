# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/pvt_tiny_seed43/pvt_tiny/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8187 |
| Accuracy | 0.8390 |
| Macro-AUROC (OvR) | 0.9509 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8388 |
| Balanced Acc | 0.8388 |
| Macro-Sensitivity | 0.8388 |
| Macro-Specificity | 0.9104 |
| Macro-Precision | 0.8247 |
| AUC-PR (Macro) | 0.8890 |
| Cohen's Kappa | 0.7279 |
| MCC | 0.7336 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 2064.0 (0.48 ms/张) |
| End-to-End FPS | 186.4 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8462 | 0.8788 | 0.8788 | 0.8462 | 0.9318 | 0.9542 | 66 |
| malignant | 0.6875 | 0.9767 | 0.9167 | 0.7857 | 0.8936 | 0.9531 | 0.8977 | 32 |
| normal | 0.9500 | 0.9082 | 0.6786 | 0.7917 | 0.9889 | 0.9679 | 0.8151 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=8 TN=44 FN=8
- **malignant**: TP=22 FP=2 TN=84 FN=10
- **normal**: TP=19 FP=9 TN=89 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 2 | 6 |
| malignant | 7 | 22 | 3 |
| normal | 1 | 0 | 19 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8788    0.8788    0.8788        66
   malignant     0.9167    0.6875    0.7857        32
      normal     0.6786    0.9500    0.7917        20

    accuracy                         0.8390       118
   macro avg     0.8247    0.8388    0.8187       118
weighted avg     0.8551    0.8390    0.8388       118

```