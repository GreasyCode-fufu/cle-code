# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/vit_base_patch16_224_seed44/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8329 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9387 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8553 |
| Balanced Acc | 0.8364 |
| Macro-Sensitivity | 0.8364 |
| Macro-Specificity | 0.9187 |
| Macro-Precision | 0.8319 |
| AUC-PR (Macro) | 0.8909 |
| Cohen's Kappa | 0.7534 |
| MCC | 0.7540 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 847.0 (1.18 ms/张) |
| End-to-End FPS | 165.5 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9091 | 0.8654 | 0.8955 | 0.9023 | 0.8824 | 0.9191 | 0.9414 | 66 |
| malignant | 0.7500 | 0.9419 | 0.8276 | 0.7869 | 0.9101 | 0.9270 | 0.8313 | 32 |
| normal | 0.8500 | 0.9490 | 0.7727 | 0.8095 | 0.9688 | 0.9699 | 0.9000 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=7 TN=45 FN=6
- **malignant**: TP=24 FP=5 TN=81 FN=8
- **normal**: TP=17 FP=5 TN=93 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 4 | 2 |
| malignant | 5 | 24 | 3 |
| normal | 2 | 1 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8955    0.9091    0.9023        66
   malignant     0.8276    0.7500    0.7869        32
      normal     0.7727    0.8500    0.8095        20

    accuracy                         0.8559       118
   macro avg     0.8319    0.8364    0.8329       118
weighted avg     0.8563    0.8559    0.8553       118

```