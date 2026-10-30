# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/vit_base_patch16_224_seed43/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8290 |
| Accuracy | 0.8621 |
| Macro-AUROC (OvR) | 0.9495 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8589 |
| Balanced Acc | 0.8271 |
| Macro-Sensitivity | 0.8271 |
| Macro-Specificity | 0.9190 |
| Macro-Precision | 0.8460 |
| AUC-PR (Macro) | 0.9047 |
| Cohen's Kappa | 0.7600 |
| MCC | 0.7651 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 841.9 (1.19 ms/张) |
| End-to-End FPS | 159.1 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9538 | 0.8431 | 0.8857 | 0.9185 | 0.9348 | 0.9641 | 0.9595 | 65 |
| malignant | 0.6774 | 0.9765 | 0.9130 | 0.7778 | 0.8925 | 0.9355 | 0.9001 | 31 |
| normal | 0.8500 | 0.9375 | 0.7391 | 0.7907 | 0.9677 | 0.9490 | 0.8544 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=8 TN=43 FN=3
- **malignant**: TP=21 FP=2 TN=83 FN=10
- **normal**: TP=17 FP=6 TN=90 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 2 | 1 |
| malignant | 5 | 21 | 5 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8857    0.9538    0.9185        65
   malignant     0.9130    0.6774    0.7778        31
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.8621       116
   macro avg     0.8460    0.8271    0.8290       116
weighted avg     0.8677    0.8621    0.8589       116

```