# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_fgc_seed42/convnext_fgc/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8468 |
| Accuracy | 0.8793 |
| Macro-AUROC (OvR) | 0.9558 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8782 |
| Balanced Acc | 0.8427 |
| Macro-Sensitivity | 0.8427 |
| Macro-Specificity | 0.9317 |
| Macro-Precision | 0.8542 |
| AUC-PR (Macro) | 0.9204 |
| Cohen's Kappa | 0.7912 |
| MCC | 0.7926 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 402.2 (2.49 ms/张) |
| End-to-End FPS | 128.2 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.8824 | 0.9118 | 0.9323 | 0.9375 | 0.9614 | 0.9625 | 65 |
| malignant | 0.7742 | 0.9647 | 0.8889 | 0.8276 | 0.9213 | 0.9476 | 0.9109 | 31 |
| normal | 0.8000 | 0.9479 | 0.7619 | 0.7805 | 0.9579 | 0.9583 | 0.8878 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=6 TN=45 FN=3
- **malignant**: TP=24 FP=3 TN=82 FN=7
- **normal**: TP=16 FP=5 TN=91 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 2 | 1 |
| malignant | 3 | 24 | 4 |
| normal | 3 | 1 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9118    0.9538    0.9323        65
   malignant     0.8889    0.7742    0.8276        31
      normal     0.7619    0.8000    0.7805        20

    accuracy                         0.8793       116
   macro avg     0.8542    0.8427    0.8468       116
weighted avg     0.8798    0.8793    0.8782       116

```