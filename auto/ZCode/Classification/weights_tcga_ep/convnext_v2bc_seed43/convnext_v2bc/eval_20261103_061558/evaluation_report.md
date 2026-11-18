# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_v2bc_seed43/convnext_v2bc/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8062 |
| Accuracy | 0.8442 |
| Macro-AUROC (OvR) | 0.9697 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8396 |
| Balanced Acc | 0.8065 |
| Macro-Sensitivity | 0.8065 |
| Macro-Specificity | 0.9673 |
| Macro-Precision | 0.8136 |
| AUC-PR (Macro) | 0.8859 |
| Cohen's Kappa | 0.7988 |
| MCC | 0.8000 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 547.8 (1.83 ms/张) |
| End-to-End FPS | 299.3 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8894 | 0.9949 | 0.7696 | 0.8252 | 0.9979 | 0.9974 | 0.9129 | 199 |
| cancer | 0.9267 | 0.9367 | 0.8289 | 0.8750 | 0.9747 | 0.9785 | 0.9519 | 2645 |
| empty | 0.9889 | 0.9649 | 0.9256 | 0.9562 | 0.9950 | 0.9870 | 0.9711 | 3256 |
| normal | 0.8012 | 0.9840 | 0.9073 | 0.8509 | 0.9621 | 0.9672 | 0.9156 | 1735 |
| other | 0.6383 | 0.9444 | 0.6884 | 0.6624 | 0.9313 | 0.9303 | 0.7817 | 1717 |
| stroma | 0.5943 | 0.9790 | 0.7618 | 0.6677 | 0.9552 | 0.9577 | 0.7819 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=177 FP=53 TN=10382 FN=22
- **cancer**: TP=2451 FP=506 TN=7483 FN=194
- **empty**: TP=3220 FP=259 TN=7119 FN=36
- **normal**: TP=1390 FP=142 TN=8757 FN=345
- **other**: TP=1096 FP=496 TN=8421 FN=621
- **stroma**: TP=643 FP=201 TN=9351 FN=439

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 177 | 6 | 1 | 4 | 10 | 1 |
| cancer | 8 | 2451 | 0 | 36 | 99 | 51 |
| empty | 0 | 0 | 3220 | 0 | 36 | 0 |
| normal | 4 | 181 | 0 | 1390 | 101 | 59 |
| other | 39 | 209 | 258 | 25 | 1096 | 90 |
| stroma | 2 | 110 | 0 | 77 | 250 | 643 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7696    0.8894    0.8252       199
      cancer     0.8289    0.9267    0.8750      2645
       empty     0.9256    0.9889    0.9562      3256
      normal     0.9073    0.8012    0.8509      1735
       other     0.6884    0.6383    0.6624      1717
      stroma     0.7618    0.5943    0.6677      1082

    accuracy                         0.8442     10634
   macro avg     0.8136    0.8065    0.8062     10634
weighted avg     0.8407    0.8442    0.8396     10634

```