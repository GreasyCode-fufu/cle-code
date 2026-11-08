# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r5/convnext_v6f_seed42/convnext_v6f/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8660 |
| Accuracy | 0.8976 |
| Macro-AUROC (OvR) | 0.9845 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8977 |
| Balanced Acc | 0.8766 |
| Macro-Sensitivity | 0.8766 |
| Macro-Specificity | 0.9796 |
| Macro-Precision | 0.8571 |
| AUC-PR (Macro) | 0.9225 |
| Cohen's Kappa | 0.8708 |
| MCC | 0.8708 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.26M |
| Pure-Inference FPS | 607.9 (1.65 ms/张) |
| End-to-End FPS | 310.8 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8700 | 0.9948 | 0.7500 | 0.8056 | 0.9977 | 0.9960 | 0.9011 | 100 |
| cancer | 0.8841 | 0.9755 | 0.9165 | 0.9000 | 0.9652 | 0.9794 | 0.9575 | 1329 |
| empty | 0.9914 | 0.9988 | 0.9969 | 0.9941 | 0.9966 | 0.9999 | 0.9999 | 1621 |
| normal | 0.9625 | 0.9853 | 0.9340 | 0.9480 | 0.9918 | 0.9965 | 0.9838 | 1014 |
| other | 0.7941 | 0.9585 | 0.7773 | 0.7856 | 0.9623 | 0.9658 | 0.8870 | 879 |
| stroma | 0.7576 | 0.9648 | 0.7677 | 0.7626 | 0.9629 | 0.9691 | 0.8060 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=87 FP=29 TN=5573 FN=13
- **cancer**: TP=1175 FP=107 TN=4266 FN=154
- **empty**: TP=1607 FP=5 TN=4076 FN=14
- **normal**: TP=976 FP=69 TN=4619 FN=38
- **other**: TP=698 FP=200 TN=4623 FN=181
- **stroma**: TP=575 FP=174 TN=4769 FN=184

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 87 | 2 | 1 | 5 | 4 | 1 |
| cancer | 16 | 1175 | 0 | 39 | 50 | 49 |
| empty | 1 | 0 | 1607 | 0 | 13 | 0 |
| normal | 2 | 14 | 0 | 976 | 11 | 11 |
| other | 5 | 55 | 4 | 4 | 698 | 113 |
| stroma | 5 | 36 | 0 | 21 | 122 | 575 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7500    0.8700    0.8056       100
      cancer     0.9165    0.8841    0.9000      1329
       empty     0.9969    0.9914    0.9941      1621
      normal     0.9340    0.9625    0.9480      1014
       other     0.7773    0.7941    0.7856       879
      stroma     0.7677    0.7576    0.7626       759

    accuracy                         0.8976      5702
   macro avg     0.8571    0.8766    0.8660      5702
weighted avg     0.8983    0.8976    0.8977      5702

```