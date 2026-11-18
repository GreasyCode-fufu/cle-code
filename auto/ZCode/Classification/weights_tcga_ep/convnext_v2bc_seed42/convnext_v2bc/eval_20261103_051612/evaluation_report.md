# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_v2bc_seed42/convnext_v2bc/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8103 |
| Accuracy | 0.8414 |
| Macro-AUROC (OvR) | 0.9574 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8394 |
| Balanced Acc | 0.8090 |
| Macro-Sensitivity | 0.8090 |
| Macro-Specificity | 0.9671 |
| Macro-Precision | 0.8167 |
| AUC-PR (Macro) | 0.8680 |
| Cohen's Kappa | 0.7957 |
| MCC | 0.7967 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 543.4 (1.84 ms/张) |
| End-to-End FPS | 298.6 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8744 | 0.9959 | 0.8018 | 0.8365 | 0.9976 | 0.9978 | 0.9001 | 199 |
| cancer | 0.9096 | 0.9663 | 0.8994 | 0.9045 | 0.9700 | 0.9868 | 0.9703 | 2645 |
| empty | 0.9979 | 0.9580 | 0.9129 | 0.9535 | 0.9990 | 0.9833 | 0.9585 | 3256 |
| normal | 0.7147 | 0.9838 | 0.8960 | 0.7951 | 0.9465 | 0.9664 | 0.8880 | 1735 |
| other | 0.6441 | 0.9215 | 0.6124 | 0.6279 | 0.9308 | 0.8467 | 0.6801 | 1717 |
| stroma | 0.7135 | 0.9769 | 0.7774 | 0.7441 | 0.9678 | 0.9633 | 0.8111 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=174 FP=43 TN=10392 FN=25
- **cancer**: TP=2406 FP=269 TN=7720 FN=239
- **empty**: TP=3249 FP=310 TN=7068 FN=7
- **normal**: TP=1240 FP=144 TN=8755 FN=495
- **other**: TP=1106 FP=700 TN=8217 FN=611
- **stroma**: TP=772 FP=221 TN=9331 FN=310

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 174 | 8 | 2 | 5 | 9 | 1 |
| cancer | 5 | 2406 | 9 | 28 | 186 | 11 |
| empty | 0 | 2 | 3249 | 0 | 5 | 0 |
| normal | 0 | 79 | 1 | 1240 | 321 | 94 |
| other | 35 | 133 | 297 | 31 | 1106 | 115 |
| stroma | 3 | 47 | 1 | 80 | 179 | 772 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8018    0.8744    0.8365       199
      cancer     0.8994    0.9096    0.9045      2645
       empty     0.9129    0.9979    0.9535      3256
      normal     0.8960    0.7147    0.7951      1735
       other     0.6124    0.6441    0.6279      1717
      stroma     0.7774    0.7135    0.7441      1082

    accuracy                         0.8414     10634
   macro avg     0.8167    0.8090    0.8103     10634
weighted avg     0.8424    0.8414    0.8394     10634

```