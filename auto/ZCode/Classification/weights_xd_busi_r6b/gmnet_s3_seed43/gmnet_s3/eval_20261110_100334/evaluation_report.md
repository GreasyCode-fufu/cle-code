# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/gmnet_s3_seed43/gmnet_s3/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8275 |
| Accuracy | 0.8534 |
| Macro-AUROC (OvR) | 0.9220 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8553 |
| Balanced Acc | 0.8388 |
| Macro-Sensitivity | 0.8388 |
| Macro-Specificity | 0.9212 |
| Macro-Precision | 0.8250 |
| AUC-PR (Macro) | 0.8752 |
| Cohen's Kappa | 0.7522 |
| MCC | 0.7542 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1222.9 (0.82 ms/张) |
| End-to-End FPS | 162.1 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8923 | 0.8824 | 0.9062 | 0.8992 | 0.8654 | 0.9363 | 0.9383 | 65 |
| malignant | 0.7742 | 0.9647 | 0.8889 | 0.8276 | 0.9213 | 0.9093 | 0.8863 | 31 |
| normal | 0.8500 | 0.9167 | 0.6800 | 0.7556 | 0.9670 | 0.9203 | 0.8008 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=6 TN=45 FN=7
- **malignant**: TP=24 FP=3 TN=82 FN=7
- **normal**: TP=17 FP=8 TN=88 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 3 | 4 |
| malignant | 3 | 24 | 4 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9062    0.8923    0.8992        65
   malignant     0.8889    0.7742    0.8276        31
      normal     0.6800    0.8500    0.7556        20

    accuracy                         0.8534       116
   macro avg     0.8250    0.8388    0.8275       116
weighted avg     0.8626    0.8534    0.8553       116

```