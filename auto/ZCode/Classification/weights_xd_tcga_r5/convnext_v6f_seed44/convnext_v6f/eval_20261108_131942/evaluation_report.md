# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r5/convnext_v6f_seed44/convnext_v6f/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8054 |
| Accuracy | 0.8391 |
| Macro-AUROC (OvR) | 0.9606 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8399 |
| Balanced Acc | 0.8143 |
| Macro-Sensitivity | 0.8143 |
| Macro-Specificity | 0.9674 |
| Macro-Precision | 0.8037 |
| AUC-PR (Macro) | 0.8630 |
| Cohen's Kappa | 0.7940 |
| MCC | 0.7953 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.26M |
| Pure-Inference FPS | 614.2 (1.63 ms/张) |
| End-to-End FPS | 317.2 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8794 | 0.9955 | 0.7883 | 0.8314 | 0.9977 | 0.9980 | 0.9243 | 199 |
| cancer | 0.8571 | 0.9812 | 0.9379 | 0.8957 | 0.9540 | 0.9712 | 0.9475 | 2645 |
| empty | 0.9969 | 0.9625 | 0.9214 | 0.9577 | 0.9986 | 0.9855 | 0.9664 | 3256 |
| normal | 0.7418 | 0.9847 | 0.9044 | 0.8151 | 0.9514 | 0.9603 | 0.8914 | 1735 |
| other | 0.6645 | 0.9363 | 0.6676 | 0.6661 | 0.9355 | 0.8948 | 0.7378 | 1717 |
| stroma | 0.7458 | 0.9442 | 0.6022 | 0.6664 | 0.9704 | 0.9540 | 0.7106 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=175 FP=47 TN=10388 FN=24
- **cancer**: TP=2267 FP=150 TN=7839 FN=378
- **empty**: TP=3246 FP=277 TN=7101 FN=10
- **normal**: TP=1287 FP=136 TN=8763 FN=448
- **other**: TP=1141 FP=568 TN=8349 FN=576
- **stroma**: TP=807 FP=533 TN=9019 FN=275

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 175 | 3 | 2 | 5 | 13 | 1 |
| cancer | 4 | 2267 | 0 | 55 | 251 | 68 |
| empty | 1 | 0 | 3246 | 0 | 9 | 0 |
| normal | 4 | 64 | 1 | 1287 | 111 | 268 |
| other | 32 | 54 | 274 | 20 | 1141 | 196 |
| stroma | 6 | 29 | 0 | 56 | 184 | 807 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7883    0.8794    0.8314       199
      cancer     0.9379    0.8571    0.8957      2645
       empty     0.9214    0.9969    0.9577      3256
      normal     0.9044    0.7418    0.8151      1735
       other     0.6676    0.6645    0.6661      1717
      stroma     0.6022    0.7458    0.6664      1082

    accuracy                         0.8391     10634
   macro avg     0.8037    0.8143    0.8054     10634
weighted avg     0.8468    0.8391    0.8399     10634

```