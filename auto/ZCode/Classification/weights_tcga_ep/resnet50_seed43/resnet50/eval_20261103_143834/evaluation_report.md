# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/resnet50_seed43/resnet50/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8019 |
| Accuracy | 0.8402 |
| Macro-AUROC (OvR) | 0.9601 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8369 |
| Balanced Acc | 0.8032 |
| Macro-Sensitivity | 0.8032 |
| Macro-Specificity | 0.9669 |
| Macro-Precision | 0.8059 |
| AUC-PR (Macro) | 0.8717 |
| Cohen's Kappa | 0.7941 |
| MCC | 0.7949 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.52M |
| Pure-Inference FPS | 897.7 (1.11 ms/张) |
| End-to-End FPS | 384.6 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8995 | 0.9954 | 0.7885 | 0.8404 | 0.9981 | 0.9981 | 0.9310 | 199 |
| cancer | 0.9214 | 0.9658 | 0.8993 | 0.9102 | 0.9738 | 0.9887 | 0.9713 | 2645 |
| empty | 0.9923 | 0.9570 | 0.9107 | 0.9497 | 0.9965 | 0.9909 | 0.9753 | 3256 |
| normal | 0.7666 | 0.9809 | 0.8867 | 0.8223 | 0.9557 | 0.9426 | 0.8799 | 1735 |
| other | 0.6564 | 0.9289 | 0.6400 | 0.6481 | 0.9335 | 0.9068 | 0.7316 | 1717 |
| stroma | 0.5832 | 0.9731 | 0.7106 | 0.6406 | 0.9537 | 0.9335 | 0.7407 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=179 FP=48 TN=10387 FN=20
- **cancer**: TP=2437 FP=273 TN=7716 FN=208
- **empty**: TP=3231 FP=317 TN=7061 FN=25
- **normal**: TP=1330 FP=170 TN=8729 FN=405
- **other**: TP=1127 FP=634 TN=8283 FN=590
- **stroma**: TP=631 FP=257 TN=9295 FN=451

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 179 | 4 | 2 | 5 | 8 | 1 |
| cancer | 5 | 2437 | 2 | 74 | 94 | 33 |
| empty | 0 | 8 | 3231 | 0 | 17 | 0 |
| normal | 1 | 71 | 0 | 1330 | 235 | 98 |
| other | 40 | 121 | 278 | 26 | 1127 | 125 |
| stroma | 2 | 69 | 35 | 65 | 280 | 631 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7885    0.8995    0.8404       199
      cancer     0.8993    0.9214    0.9102      2645
       empty     0.9107    0.9923    0.9497      3256
      normal     0.8867    0.7666    0.8223      1735
       other     0.6400    0.6564    0.6481      1717
      stroma     0.7106    0.5832    0.6406      1082

    accuracy                         0.8402     10634
   macro avg     0.8059    0.8032    0.8019     10634
weighted avg     0.8376    0.8402    0.8369     10634

```