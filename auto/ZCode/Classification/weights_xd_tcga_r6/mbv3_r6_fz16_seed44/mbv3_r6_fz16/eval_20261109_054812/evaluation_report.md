# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_r6_fz16_seed44/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7871 |
| Accuracy | 0.8330 |
| Macro-AUROC (OvR) | 0.9509 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8285 |
| Balanced Acc | 0.7910 |
| Macro-Sensitivity | 0.7910 |
| Macro-Specificity | 0.9653 |
| Macro-Precision | 0.7912 |
| AUC-PR (Macro) | 0.8411 |
| Cohen's Kappa | 0.7845 |
| MCC | 0.7856 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.92M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 9660.4 (0.10 ms/张) |
| End-to-End FPS | 626.1 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8844 | 0.9940 | 0.7364 | 0.8037 | 0.9978 | 0.9940 | 0.9029 | 199 |
| cancer | 0.9225 | 0.9522 | 0.8646 | 0.8926 | 0.9738 | 0.9860 | 0.9608 | 2645 |
| empty | 0.9951 | 0.9602 | 0.9168 | 0.9543 | 0.9977 | 0.9881 | 0.9575 | 3256 |
| normal | 0.7625 | 0.9830 | 0.8976 | 0.8246 | 0.9550 | 0.9541 | 0.8830 | 1735 |
| other | 0.6313 | 0.9293 | 0.6324 | 0.6319 | 0.9290 | 0.8652 | 0.6668 | 1717 |
| stroma | 0.5499 | 0.9732 | 0.6992 | 0.6156 | 0.9502 | 0.9182 | 0.6756 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=176 FP=63 TN=10372 FN=23
- **cancer**: TP=2440 FP=382 TN=7607 FN=205
- **empty**: TP=3240 FP=294 TN=7084 FN=16
- **normal**: TP=1323 FP=151 TN=8748 FN=412
- **other**: TP=1084 FP=630 TN=8287 FN=633
- **stroma**: TP=595 FP=256 TN=9296 FN=487

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 176 | 3 | 2 | 5 | 11 | 2 |
| cancer | 6 | 2440 | 1 | 29 | 109 | 60 |
| empty | 0 | 1 | 3240 | 0 | 15 | 0 |
| normal | 1 | 156 | 4 | 1323 | 185 | 66 |
| other | 52 | 143 | 287 | 23 | 1084 | 128 |
| stroma | 4 | 79 | 0 | 94 | 310 | 595 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7364    0.8844    0.8037       199
      cancer     0.8646    0.9225    0.8926      2645
       empty     0.9168    0.9951    0.9543      3256
      normal     0.8976    0.7625    0.8246      1735
       other     0.6324    0.6313    0.6319      1717
      stroma     0.6992    0.5499    0.6156      1082

    accuracy                         0.8330     10634
   macro avg     0.7912    0.7910    0.7871     10634
weighted avg     0.8293    0.8330    0.8285     10634

```