# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_tiny_seed43/convnext_tiny/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8473 |
| Accuracy | 0.8797 |
| Macro-AUROC (OvR) | 0.9787 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8811 |
| Balanced Acc | 0.8454 |
| Macro-Sensitivity | 0.8454 |
| Macro-Specificity | 0.9759 |
| Macro-Precision | 0.8565 |
| AUC-PR (Macro) | 0.9051 |
| Cohen's Kappa | 0.8482 |
| MCC | 0.8495 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 595.3 (1.68 ms/张) |
| End-to-End FPS | 314.8 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7600 | 0.9968 | 0.8085 | 0.7835 | 0.9957 | 0.9919 | 0.8697 | 100 |
| cancer | 0.8751 | 0.9662 | 0.8871 | 0.8811 | 0.9622 | 0.9729 | 0.9449 | 1329 |
| empty | 0.9938 | 0.9951 | 0.9877 | 0.9908 | 0.9975 | 0.9996 | 0.9991 | 1621 |
| normal | 0.8866 | 0.9859 | 0.9316 | 0.9085 | 0.9757 | 0.9906 | 0.9678 | 1014 |
| other | 0.7110 | 0.9809 | 0.8717 | 0.7832 | 0.9490 | 0.9547 | 0.8814 | 879 |
| stroma | 0.8458 | 0.9308 | 0.6524 | 0.7367 | 0.9752 | 0.9626 | 0.7679 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=76 FP=18 TN=5584 FN=24
- **cancer**: TP=1163 FP=148 TN=4225 FN=166
- **empty**: TP=1611 FP=20 TN=4061 FN=10
- **normal**: TP=899 FP=66 TN=4622 FN=115
- **other**: TP=625 FP=92 TN=4731 FN=254
- **stroma**: TP=642 FP=342 TN=4601 FN=117

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 76 | 5 | 2 | 4 | 8 | 5 |
| cancer | 14 | 1163 | 1 | 14 | 23 | 114 |
| empty | 0 | 2 | 1611 | 0 | 6 | 2 |
| normal | 1 | 45 | 0 | 899 | 6 | 63 |
| other | 1 | 59 | 17 | 19 | 625 | 158 |
| stroma | 2 | 37 | 0 | 29 | 49 | 642 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8085    0.7600    0.7835       100
      cancer     0.8871    0.8751    0.8811      1329
       empty     0.9877    0.9938    0.9908      1621
      normal     0.9316    0.8866    0.9085      1014
       other     0.8717    0.7110    0.7832       879
      stroma     0.6524    0.8458    0.7367       759

    accuracy                         0.8797      5702
   macro avg     0.8565    0.8454    0.8473      5702
weighted avg     0.8886    0.8797    0.8811      5702

```