# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_fgc_seed43/convnext_fgc/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7850 |
| Accuracy | 0.8051 |
| Macro-AUROC (OvR) | 0.9310 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8097 |
| Balanced Acc | 0.8177 |
| Macro-Sensitivity | 0.8177 |
| Macro-Specificity | 0.9013 |
| Macro-Precision | 0.7753 |
| AUC-PR (Macro) | 0.8728 |
| Cohen's Kappa | 0.6808 |
| MCC | 0.6875 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 359.6 (2.78 ms/张) |
| End-to-End FPS | 128.4 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8030 | 0.8846 | 0.8983 | 0.8480 | 0.7797 | 0.9053 | 0.9303 | 66 |
| malignant | 0.7500 | 0.9419 | 0.8276 | 0.7869 | 0.9101 | 0.9215 | 0.7949 | 32 |
| normal | 0.9000 | 0.8776 | 0.6000 | 0.7200 | 0.9773 | 0.9663 | 0.8933 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=53 FP=6 TN=46 FN=13
- **malignant**: TP=24 FP=5 TN=81 FN=8
- **normal**: TP=18 FP=12 TN=86 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 53 | 5 | 8 |
| malignant | 4 | 24 | 4 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8983    0.8030    0.8480        66
   malignant     0.8276    0.7500    0.7869        32
      normal     0.6000    0.9000    0.7200        20

    accuracy                         0.8051       118
   macro avg     0.7753    0.8177    0.7850       118
weighted avg     0.8286    0.8051    0.8097       118

```