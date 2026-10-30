# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/fast_t0_seed44/fast_t0/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8063 |
| Accuracy | 0.8051 |
| Macro-AUROC (OvR) | 0.9478 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8077 |
| Balanced Acc | 0.8284 |
| Macro-Sensitivity | 0.8284 |
| Macro-Specificity | 0.8975 |
| Macro-Precision | 0.7932 |
| AUC-PR (Macro) | 0.9123 |
| Cohen's Kappa | 0.6797 |
| MCC | 0.6855 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 2853.9 (0.35 ms/张) |
| End-to-End FPS | 189.1 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.7727 | 0.8846 | 0.8947 | 0.8293 | 0.7541 | 0.9170 | 0.9425 | 66 |
| malignant | 0.8125 | 0.8488 | 0.6667 | 0.7324 | 0.9241 | 0.9444 | 0.8678 | 32 |
| normal | 0.9000 | 0.9592 | 0.8182 | 0.8571 | 0.9792 | 0.9821 | 0.9268 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=51 FP=6 TN=46 FN=15
- **malignant**: TP=26 FP=13 TN=73 FN=6
- **normal**: TP=18 FP=4 TN=94 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 51 | 12 | 3 |
| malignant | 5 | 26 | 1 |
| normal | 1 | 1 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8947    0.7727    0.8293        66
   malignant     0.6667    0.8125    0.7324        32
      normal     0.8182    0.9000    0.8571        20

    accuracy                         0.8051       118
   macro avg     0.7932    0.8284    0.8063       118
weighted avg     0.8199    0.8051    0.8077       118

```