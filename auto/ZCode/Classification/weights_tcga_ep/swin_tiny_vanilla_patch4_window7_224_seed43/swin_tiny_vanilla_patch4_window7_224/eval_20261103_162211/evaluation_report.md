# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/swin_tiny_vanilla_patch4_window7_224_seed43/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8247 |
| Accuracy | 0.8499 |
| Macro-AUROC (OvR) | 0.9705 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8534 |
| Balanced Acc | 0.8324 |
| Macro-Sensitivity | 0.8324 |
| Macro-Specificity | 0.9702 |
| Macro-Precision | 0.8257 |
| AUC-PR (Macro) | 0.8842 |
| Cohen's Kappa | 0.8110 |
| MCC | 0.8126 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 644.5 (1.55 ms/张) |
| End-to-End FPS | 324.7 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8600 | 0.9959 | 0.7890 | 0.8230 | 0.9975 | 0.9975 | 0.9106 | 100 |
| cancer | 0.8450 | 0.9652 | 0.8808 | 0.8625 | 0.9535 | 0.9581 | 0.9202 | 1329 |
| empty | 0.9944 | 0.9978 | 0.9944 | 0.9944 | 0.9978 | 0.9998 | 0.9995 | 1621 |
| normal | 0.7416 | 0.9919 | 0.9519 | 0.8337 | 0.9467 | 0.9821 | 0.9375 | 1014 |
| other | 0.7827 | 0.9494 | 0.7382 | 0.7598 | 0.9600 | 0.9518 | 0.8540 | 879 |
| stroma | 0.7708 | 0.9211 | 0.6000 | 0.6747 | 0.9632 | 0.9339 | 0.6834 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=86 FP=23 TN=5579 FN=14
- **cancer**: TP=1123 FP=152 TN=4221 FN=206
- **empty**: TP=1612 FP=9 TN=4072 FN=9
- **normal**: TP=752 FP=38 TN=4650 FN=262
- **other**: TP=688 FP=244 TN=4579 FN=191
- **stroma**: TP=585 FP=390 TN=4553 FN=174

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 86 | 2 | 0 | 0 | 5 | 7 |
| cancer | 12 | 1123 | 0 | 26 | 9 | 159 |
| empty | 0 | 0 | 1612 | 0 | 8 | 1 |
| normal | 6 | 84 | 0 | 752 | 85 | 87 |
| other | 1 | 43 | 9 | 2 | 688 | 136 |
| stroma | 4 | 23 | 0 | 10 | 137 | 585 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7890    0.8600    0.8230       100
      cancer     0.8808    0.8450    0.8625      1329
       empty     0.9944    0.9944    0.9944      1621
      normal     0.9519    0.7416    0.8337      1014
       other     0.7382    0.7827    0.7598       879
      stroma     0.6000    0.7708    0.6747       759

    accuracy                         0.8499      5702
   macro avg     0.8257    0.8324    0.8247      5702
weighted avg     0.8648    0.8499    0.8534      5702

```