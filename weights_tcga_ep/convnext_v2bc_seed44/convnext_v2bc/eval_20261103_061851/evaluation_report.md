# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_v2bc_seed44/convnext_v2bc/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8038 |
| Accuracy | 0.8397 |
| Macro-AUROC (OvR) | 0.9567 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8346 |
| Balanced Acc | 0.8051 |
| Macro-Sensitivity | 0.8051 |
| Macro-Specificity | 0.9662 |
| Macro-Precision | 0.8100 |
| AUC-PR (Macro) | 0.8730 |
| Cohen's Kappa | 0.7929 |
| MCC | 0.7946 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 543.7 (1.84 ms/张) |
| End-to-End FPS | 298.3 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8794 | 0.9947 | 0.7609 | 0.8159 | 0.9977 | 0.9976 | 0.9161 | 199 |
| cancer | 0.9221 | 0.9317 | 0.8171 | 0.8664 | 0.9731 | 0.9801 | 0.9475 | 2645 |
| empty | 0.9960 | 0.9603 | 0.9171 | 0.9549 | 0.9982 | 0.9832 | 0.9508 | 3256 |
| normal | 0.7366 | 0.9818 | 0.8875 | 0.8050 | 0.9503 | 0.9228 | 0.8659 | 1735 |
| other | 0.6162 | 0.9533 | 0.7178 | 0.6631 | 0.9281 | 0.9050 | 0.7581 | 1717 |
| stroma | 0.6802 | 0.9756 | 0.7595 | 0.7177 | 0.9642 | 0.9519 | 0.7994 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=175 FP=55 TN=10380 FN=24
- **cancer**: TP=2439 FP=546 TN=7443 FN=206
- **empty**: TP=3243 FP=293 TN=7085 FN=13
- **normal**: TP=1278 FP=162 TN=8737 FN=457
- **other**: TP=1058 FP=416 TN=8501 FN=659
- **stroma**: TP=736 FP=233 TN=9319 FN=346

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 175 | 7 | 2 | 4 | 10 | 1 |
| cancer | 5 | 2439 | 3 | 49 | 115 | 34 |
| empty | 0 | 0 | 3243 | 0 | 13 | 0 |
| normal | 2 | 216 | 0 | 1278 | 131 | 108 |
| other | 44 | 205 | 287 | 33 | 1058 | 90 |
| stroma | 4 | 118 | 1 | 76 | 147 | 736 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7609    0.8794    0.8159       199
      cancer     0.8171    0.9221    0.8664      2645
       empty     0.9171    0.9960    0.9549      3256
      normal     0.8875    0.7366    0.8050      1735
       other     0.7178    0.6162    0.6631      1717
      stroma     0.7595    0.6802    0.7177      1082

    accuracy                         0.8397     10634
   macro avg     0.8100    0.8051    0.8038     10634
weighted avg     0.8363    0.8397    0.8346     10634

```