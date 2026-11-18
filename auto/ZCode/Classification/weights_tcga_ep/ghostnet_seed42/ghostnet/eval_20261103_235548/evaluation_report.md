# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/ghostnet_seed42/ghostnet/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7964 |
| Accuracy | 0.8307 |
| Macro-AUROC (OvR) | 0.9601 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8281 |
| Balanced Acc | 0.7979 |
| Macro-Sensitivity | 0.7979 |
| Macro-Specificity | 0.9651 |
| Macro-Precision | 0.7977 |
| AUC-PR (Macro) | 0.8597 |
| Cohen's Kappa | 0.7822 |
| MCC | 0.7828 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.59M |
| FLOPs | 309.17M |
| Params | 3.91M |
| Pure-Inference FPS | 5466.7 (0.18 ms/张) |
| End-to-End FPS | 579.7 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8995 | 0.9964 | 0.8249 | 0.8606 | 0.9981 | 0.9957 | 0.9311 | 199 |
| cancer | 0.8858 | 0.9592 | 0.8779 | 0.8818 | 0.9621 | 0.9850 | 0.9612 | 2645 |
| empty | 0.9957 | 0.9607 | 0.9179 | 0.9552 | 0.9980 | 0.9858 | 0.9535 | 3256 |
| normal | 0.7429 | 0.9784 | 0.8704 | 0.8016 | 0.9513 | 0.9677 | 0.8908 | 1735 |
| other | 0.6523 | 0.9360 | 0.6623 | 0.6573 | 0.9332 | 0.8835 | 0.7194 | 1717 |
| stroma | 0.6109 | 0.9599 | 0.6331 | 0.6218 | 0.9561 | 0.9430 | 0.7021 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=179 FP=38 TN=10397 FN=20
- **cancer**: TP=2343 FP=326 TN=7663 FN=302
- **empty**: TP=3242 FP=290 TN=7088 FN=14
- **normal**: TP=1289 FP=192 TN=8707 FN=446
- **other**: TP=1120 FP=571 TN=8346 FN=597
- **stroma**: TP=661 FP=383 TN=9169 FN=421

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 179 | 3 | 4 | 6 | 6 | 1 |
| cancer | 3 | 2343 | 1 | 85 | 118 | 95 |
| empty | 0 | 4 | 3242 | 0 | 10 | 0 |
| normal | 2 | 129 | 3 | 1289 | 160 | 152 |
| other | 29 | 125 | 282 | 26 | 1120 | 135 |
| stroma | 4 | 65 | 0 | 75 | 277 | 661 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8249    0.8995    0.8606       199
      cancer     0.8779    0.8858    0.8818      2645
       empty     0.9179    0.9957    0.9552      3256
      normal     0.8704    0.7429    0.8016      1735
       other     0.6623    0.6523    0.6573      1717
      stroma     0.6331    0.6109    0.6218      1082

    accuracy                         0.8307     10634
   macro avg     0.7977    0.7979    0.7964     10634
weighted avg     0.8282    0.8307    0.8281     10634

```