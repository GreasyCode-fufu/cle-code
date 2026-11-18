# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r5/convnext_v6f_seed42/convnext_v6f/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8208 |
| Accuracy | 0.8390 |
| Macro-AUROC (OvR) | 0.9317 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8411 |
| Balanced Acc | 0.8370 |
| Macro-Sensitivity | 0.8370 |
| Macro-Specificity | 0.9124 |
| Macro-Precision | 0.8119 |
| AUC-PR (Macro) | 0.8896 |
| Cohen's Kappa | 0.7301 |
| MCC | 0.7319 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 830.4 (1.20 ms/张) |
| End-to-End FPS | 160.3 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.8654 | 0.8889 | 0.8682 | 0.8182 | 0.9200 | 0.9405 | 66 |
| malignant | 0.8125 | 0.9535 | 0.8667 | 0.8387 | 0.9318 | 0.9553 | 0.9060 | 32 |
| normal | 0.8500 | 0.9184 | 0.6800 | 0.7556 | 0.9677 | 0.9199 | 0.8224 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=7 TN=45 FN=10
- **malignant**: TP=26 FP=4 TN=82 FN=6
- **normal**: TP=17 FP=8 TN=90 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 4 | 6 |
| malignant | 4 | 26 | 2 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8889    0.8485    0.8682        66
   malignant     0.8667    0.8125    0.8387        32
      normal     0.6800    0.8500    0.7556        20

    accuracy                         0.8390       118
   macro avg     0.8119    0.8370    0.8208       118
weighted avg     0.8475    0.8390    0.8411       118

```