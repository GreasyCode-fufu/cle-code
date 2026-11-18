# convnext_dph 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_dph_seed42/convnext_dph/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8493 |
| Accuracy | 0.8793 |
| Macro-AUROC (OvR) | 0.9435 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8766 |
| Balanced Acc | 0.8371 |
| Macro-Sensitivity | 0.8371 |
| Macro-Specificity | 0.9260 |
| Macro-Precision | 0.8691 |
| AUC-PR (Macro) | 0.8854 |
| Cohen's Kappa | 0.7882 |
| MCC | 0.7923 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 556.4 (1.80 ms/张) |
| End-to-End FPS | 141.0 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9692 | 0.8431 | 0.8873 | 0.9265 | 0.9556 | 0.9632 | 0.9459 | 65 |
| malignant | 0.7419 | 0.9765 | 0.9200 | 0.8214 | 0.9121 | 0.9340 | 0.9057 | 31 |
| normal | 0.8000 | 0.9583 | 0.8000 | 0.8000 | 0.9583 | 0.9333 | 0.8045 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=63 FP=8 TN=43 FN=2
- **malignant**: TP=23 FP=2 TN=83 FN=8
- **normal**: TP=16 FP=4 TN=92 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 63 | 2 | 0 |
| malignant | 4 | 23 | 4 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8873    0.9692    0.9265        65
   malignant     0.9200    0.7419    0.8214        31
      normal     0.8000    0.8000    0.8000        20

    accuracy                         0.8793       116
   macro avg     0.8691    0.8371    0.8493       116
weighted avg     0.8810    0.8793    0.8766       116

```