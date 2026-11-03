# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_small_seed44/convnext_small/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8561 |
| Accuracy | 0.8874 |
| Macro-AUROC (OvR) | 0.9768 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8893 |
| Balanced Acc | 0.8676 |
| Macro-Sensitivity | 0.8676 |
| Macro-Specificity | 0.9778 |
| Macro-Precision | 0.8507 |
| AUC-PR (Macro) | 0.9038 |
| Cohen's Kappa | 0.8581 |
| MCC | 0.8591 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.42M |
| Pure-Inference FPS | 381.0 (2.62 ms/张) |
| End-to-End FPS | 242.7 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8600 | 0.9948 | 0.7478 | 0.8000 | 0.9975 | 0.9966 | 0.8967 | 100 |
| cancer | 0.8804 | 0.9753 | 0.9155 | 0.8976 | 0.9641 | 0.9794 | 0.9531 | 1329 |
| empty | 0.9932 | 0.9990 | 0.9975 | 0.9954 | 0.9973 | 0.9999 | 0.9997 | 1621 |
| normal | 0.9053 | 0.9923 | 0.9623 | 0.9329 | 0.9798 | 0.9965 | 0.9837 | 1014 |
| other | 0.7224 | 0.9699 | 0.8141 | 0.7655 | 0.9504 | 0.9260 | 0.8422 | 879 |
| stroma | 0.8445 | 0.9353 | 0.6670 | 0.7453 | 0.9751 | 0.9626 | 0.7474 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=86 FP=29 TN=5573 FN=14
- **cancer**: TP=1170 FP=108 TN=4265 FN=159
- **empty**: TP=1610 FP=4 TN=4077 FN=11
- **normal**: TP=918 FP=36 TN=4652 FN=96
- **other**: TP=635 FP=145 TN=4678 FN=244
- **stroma**: TP=641 FP=320 TN=4623 FN=118

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 86 | 1 | 0 | 2 | 9 | 2 |
| cancer | 23 | 1170 | 0 | 15 | 35 | 86 |
| empty | 0 | 0 | 1610 | 0 | 9 | 2 |
| normal | 1 | 25 | 0 | 918 | 32 | 38 |
| other | 1 | 41 | 4 | 6 | 635 | 192 |
| stroma | 4 | 41 | 0 | 13 | 60 | 641 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7478    0.8600    0.8000       100
      cancer     0.9155    0.8804    0.8976      1329
       empty     0.9975    0.9932    0.9954      1621
      normal     0.9623    0.9053    0.9329      1014
       other     0.8141    0.7224    0.7655       879
      stroma     0.6670    0.8445    0.7453       759

    accuracy                         0.8874      5702
   macro avg     0.8507    0.8676    0.8561      5702
weighted avg     0.8955    0.8874    0.8893      5702

```