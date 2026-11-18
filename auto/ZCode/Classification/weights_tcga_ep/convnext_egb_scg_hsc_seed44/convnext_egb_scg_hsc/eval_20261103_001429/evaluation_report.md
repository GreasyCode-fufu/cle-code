# convnext_egb_scg_hsc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_egb_scg_hsc_seed44/convnext_egb_scg_hsc/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8339 |
| Accuracy | 0.8764 |
| Macro-AUROC (OvR) | 0.9773 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8791 |
| Balanced Acc | 0.8630 |
| Macro-Sensitivity | 0.8630 |
| Macro-Specificity | 0.9758 |
| Macro-Precision | 0.8185 |
| AUC-PR (Macro) | 0.8976 |
| Cohen's Kappa | 0.8447 |
| MCC | 0.8461 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.58M |
| Pure-Inference FPS | 1053.8 (0.95 ms/张) |
| End-to-End FPS | 412.4 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8600 | 0.9893 | 0.5890 | 0.6992 | 0.9975 | 0.9945 | 0.8617 | 100 |
| cancer | 0.7840 | 0.9810 | 0.9262 | 0.8492 | 0.9373 | 0.9743 | 0.9358 | 1329 |
| empty | 0.9944 | 0.9988 | 0.9969 | 0.9957 | 0.9978 | 0.9998 | 0.9997 | 1621 |
| normal | 0.9349 | 0.9947 | 0.9743 | 0.9542 | 0.9860 | 0.9964 | 0.9880 | 1014 |
| other | 0.7588 | 0.9540 | 0.7503 | 0.7545 | 0.9560 | 0.9364 | 0.8450 | 879 |
| stroma | 0.8458 | 0.9373 | 0.6744 | 0.7504 | 0.9754 | 0.9626 | 0.7553 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=86 FP=60 TN=5542 FN=14
- **cancer**: TP=1042 FP=83 TN=4290 FN=287
- **empty**: TP=1612 FP=5 TN=4076 FN=9
- **normal**: TP=948 FP=25 TN=4663 FN=66
- **other**: TP=667 FP=222 TN=4601 FN=212
- **stroma**: TP=642 FP=310 TN=4633 FN=117

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 86 | 0 | 0 | 2 | 10 | 2 |
| cancer | 33 | 1042 | 0 | 1 | 140 | 113 |
| empty | 1 | 0 | 1612 | 0 | 6 | 2 |
| normal | 2 | 19 | 0 | 948 | 5 | 40 |
| other | 6 | 37 | 5 | 11 | 667 | 153 |
| stroma | 18 | 27 | 0 | 11 | 61 | 642 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.5890    0.8600    0.6992       100
      cancer     0.9262    0.7840    0.8492      1329
       empty     0.9969    0.9944    0.9957      1621
      normal     0.9743    0.9349    0.9542      1014
       other     0.7503    0.7588    0.7545       879
      stroma     0.6744    0.8458    0.7504       759

    accuracy                         0.8764      5702
   macro avg     0.8185    0.8630    0.8339      5702
weighted avg     0.8883    0.8764    0.8791      5702

```