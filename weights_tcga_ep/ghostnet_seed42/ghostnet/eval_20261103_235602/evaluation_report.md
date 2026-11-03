# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/ghostnet_seed42/ghostnet/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8546 |
| Accuracy | 0.8734 |
| Macro-AUROC (OvR) | 0.9809 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8764 |
| Balanced Acc | 0.8663 |
| Macro-Sensitivity | 0.8663 |
| Macro-Specificity | 0.9751 |
| Macro-Precision | 0.8488 |
| AUC-PR (Macro) | 0.9116 |
| Cohen's Kappa | 0.8408 |
| MCC | 0.8421 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.59M |
| FLOPs | 309.17M |
| Params | 3.91M |
| Pure-Inference FPS | 4813.5 (0.21 ms/张) |
| End-to-End FPS | 573.6 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.9100 | 0.9964 | 0.8198 | 0.8626 | 0.9984 | 0.9985 | 0.9346 | 100 |
| cancer | 0.7983 | 0.9858 | 0.9448 | 0.8654 | 0.9415 | 0.9797 | 0.9522 | 1329 |
| empty | 0.9951 | 0.9983 | 0.9957 | 0.9954 | 0.9980 | 0.9997 | 0.9995 | 1621 |
| normal | 0.8905 | 0.9893 | 0.9475 | 0.9181 | 0.9766 | 0.9937 | 0.9764 | 1014 |
| other | 0.7895 | 0.9355 | 0.6905 | 0.7367 | 0.9606 | 0.9508 | 0.8325 | 879 |
| stroma | 0.8142 | 0.9450 | 0.6944 | 0.7495 | 0.9707 | 0.9627 | 0.7743 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=91 FP=20 TN=5582 FN=9
- **cancer**: TP=1061 FP=62 TN=4311 FN=268
- **empty**: TP=1613 FP=7 TN=4074 FN=8
- **normal**: TP=903 FP=50 TN=4638 FN=111
- **other**: TP=694 FP=311 TN=4512 FN=185
- **stroma**: TP=618 FP=272 TN=4671 FN=141

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 91 | 0 | 0 | 1 | 3 | 5 |
| cancer | 11 | 1061 | 0 | 29 | 153 | 75 |
| empty | 0 | 1 | 1613 | 0 | 5 | 2 |
| normal | 3 | 19 | 0 | 903 | 45 | 44 |
| other | 5 | 24 | 7 | 3 | 694 | 146 |
| stroma | 1 | 18 | 0 | 17 | 105 | 618 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8198    0.9100    0.8626       100
      cancer     0.9448    0.7983    0.8654      1329
       empty     0.9957    0.9951    0.9954      1621
      normal     0.9475    0.8905    0.9181      1014
       other     0.6905    0.7895    0.7367       879
      stroma     0.6944    0.8142    0.7495       759

    accuracy                         0.8734      5702
   macro avg     0.8488    0.8663    0.8546      5702
weighted avg     0.8850    0.8734    0.8764      5702

```