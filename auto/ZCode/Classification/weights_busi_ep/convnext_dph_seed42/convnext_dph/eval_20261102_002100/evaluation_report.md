# convnext_dph 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_dph_seed42/convnext_dph/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8158 |
| Accuracy | 0.8305 |
| Macro-AUROC (OvR) | 0.9420 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8302 |
| Balanced Acc | 0.8212 |
| Macro-Sensitivity | 0.8212 |
| Macro-Specificity | 0.9020 |
| Macro-Precision | 0.8121 |
| AUC-PR (Macro) | 0.8882 |
| Cohen's Kappa | 0.7111 |
| MCC | 0.7114 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 405.0 (2.47 ms/张) |
| End-to-End FPS | 133.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8636 | 0.8269 | 0.8636 | 0.8636 | 0.8269 | 0.9202 | 0.9445 | 66 |
| malignant | 0.7500 | 0.9302 | 0.8000 | 0.7742 | 0.9091 | 0.9360 | 0.8282 | 32 |
| normal | 0.8500 | 0.9490 | 0.7727 | 0.8095 | 0.9688 | 0.9699 | 0.8919 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=9 TN=43 FN=9
- **malignant**: TP=24 FP=6 TN=80 FN=8
- **normal**: TP=17 FP=5 TN=93 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 6 | 3 |
| malignant | 6 | 24 | 2 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8636    0.8636    0.8636        66
   malignant     0.8000    0.7500    0.7742        32
      normal     0.7727    0.8500    0.8095        20

    accuracy                         0.8305       118
   macro avg     0.8121    0.8212    0.8158       118
weighted avg     0.8310    0.8305    0.8302       118

```