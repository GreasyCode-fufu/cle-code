# convnext_dph 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_dph_seed44/convnext_dph/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8828 |
| Accuracy | 0.9052 |
| Macro-AUROC (OvR) | 0.9459 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9030 |
| Balanced Acc | 0.8696 |
| Macro-Sensitivity | 0.8696 |
| Macro-Specificity | 0.9399 |
| Macro-Precision | 0.9038 |
| AUC-PR (Macro) | 0.9020 |
| Cohen's Kappa | 0.8336 |
| MCC | 0.8379 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 470.1 (2.13 ms/张) |
| End-to-End FPS | 132.5 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9846 | 0.8627 | 0.9014 | 0.9412 | 0.9778 | 0.9641 | 0.9407 | 65 |
| malignant | 0.7742 | 0.9882 | 0.9600 | 0.8571 | 0.9231 | 0.9324 | 0.9179 | 31 |
| normal | 0.8500 | 0.9688 | 0.8500 | 0.8500 | 0.9688 | 0.9411 | 0.8473 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=64 FP=7 TN=44 FN=1
- **malignant**: TP=24 FP=1 TN=84 FN=7
- **normal**: TP=17 FP=3 TN=93 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 64 | 1 | 0 |
| malignant | 4 | 24 | 3 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9014    0.9846    0.9412        65
   malignant     0.9600    0.7742    0.8571        31
      normal     0.8500    0.8500    0.8500        20

    accuracy                         0.9052       116
   macro avg     0.9038    0.8696    0.8828       116
weighted avg     0.9082    0.9052    0.9030       116

```