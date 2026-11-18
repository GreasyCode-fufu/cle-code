# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_v2bc_seed43/convnext_v2bc/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8606 |
| Accuracy | 0.8916 |
| Macro-AUROC (OvR) | 0.9787 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8916 |
| Balanced Acc | 0.8580 |
| Macro-Sensitivity | 0.8580 |
| Macro-Specificity | 0.9783 |
| Macro-Precision | 0.8635 |
| AUC-PR (Macro) | 0.9087 |
| Cohen's Kappa | 0.8631 |
| MCC | 0.8631 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 537.6 (1.86 ms/张) |
| End-to-End FPS | 297.5 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8000 | 0.9973 | 0.8421 | 0.8205 | 0.9964 | 0.9896 | 0.8740 | 100 |
| cancer | 0.9007 | 0.9691 | 0.8986 | 0.8997 | 0.9698 | 0.9793 | 0.9535 | 1329 |
| empty | 0.9877 | 0.9993 | 0.9981 | 0.9929 | 0.9951 | 0.9996 | 0.9995 | 1621 |
| normal | 0.9606 | 0.9861 | 0.9374 | 0.9489 | 0.9914 | 0.9955 | 0.9844 | 1014 |
| other | 0.7838 | 0.9623 | 0.7910 | 0.7874 | 0.9607 | 0.9465 | 0.8732 | 879 |
| stroma | 0.7154 | 0.9559 | 0.7135 | 0.7145 | 0.9563 | 0.9615 | 0.7676 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=80 FP=15 TN=5587 FN=20
- **cancer**: TP=1197 FP=135 TN=4238 FN=132
- **empty**: TP=1601 FP=3 TN=4078 FN=20
- **normal**: TP=974 FP=65 TN=4623 FN=40
- **other**: TP=689 FP=182 TN=4641 FN=190
- **stroma**: TP=543 FP=218 TN=4725 FN=216

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 80 | 2 | 0 | 3 | 11 | 4 |
| cancer | 8 | 1197 | 0 | 25 | 18 | 81 |
| empty | 0 | 0 | 1601 | 0 | 18 | 2 |
| normal | 2 | 18 | 0 | 974 | 9 | 11 |
| other | 1 | 55 | 3 | 11 | 689 | 120 |
| stroma | 4 | 60 | 0 | 26 | 126 | 543 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8421    0.8000    0.8205       100
      cancer     0.8986    0.9007    0.8997      1329
       empty     0.9981    0.9877    0.9929      1621
      normal     0.9374    0.9606    0.9489      1014
       other     0.7910    0.7838    0.7874       879
      stroma     0.7135    0.7154    0.7145       759

    accuracy                         0.8916      5702
   macro avg     0.8635    0.8580    0.8606      5702
weighted avg     0.8916    0.8916    0.8916      5702

```