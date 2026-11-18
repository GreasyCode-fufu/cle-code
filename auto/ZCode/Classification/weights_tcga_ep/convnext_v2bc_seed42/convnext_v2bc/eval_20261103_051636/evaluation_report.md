# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_v2bc_seed42/convnext_v2bc/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8349 |
| Accuracy | 0.8622 |
| Macro-AUROC (OvR) | 0.9758 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8633 |
| Balanced Acc | 0.8500 |
| Macro-Sensitivity | 0.8500 |
| Macro-Specificity | 0.9723 |
| Macro-Precision | 0.8242 |
| AUC-PR (Macro) | 0.9014 |
| Cohen's Kappa | 0.8261 |
| MCC | 0.8265 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 537.9 (1.86 ms/张) |
| End-to-End FPS | 297.8 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8900 | 0.9938 | 0.7177 | 0.7946 | 0.9980 | 0.9967 | 0.8976 | 100 |
| cancer | 0.8111 | 0.9600 | 0.8603 | 0.8350 | 0.9436 | 0.9647 | 0.9162 | 1329 |
| empty | 0.9981 | 0.9917 | 0.9794 | 0.9887 | 0.9993 | 0.9993 | 0.9981 | 1621 |
| normal | 0.8698 | 0.9902 | 0.9504 | 0.9083 | 0.9724 | 0.9941 | 0.9757 | 1014 |
| other | 0.7258 | 0.9378 | 0.6802 | 0.7023 | 0.9494 | 0.9287 | 0.8045 | 879 |
| stroma | 0.8050 | 0.9603 | 0.7571 | 0.7803 | 0.9698 | 0.9710 | 0.8164 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=89 FP=35 TN=5567 FN=11
- **cancer**: TP=1078 FP=175 TN=4198 FN=251
- **empty**: TP=1618 FP=34 TN=4047 FN=3
- **normal**: TP=882 FP=46 TN=4642 FN=132
- **other**: TP=638 FP=300 TN=4523 FN=241
- **stroma**: TP=611 FP=196 TN=4747 FN=148

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 89 | 0 | 3 | 4 | 3 | 1 |
| cancer | 26 | 1078 | 1 | 8 | 175 | 41 |
| empty | 0 | 0 | 1618 | 0 | 3 | 0 |
| normal | 3 | 33 | 0 | 882 | 72 | 24 |
| other | 1 | 71 | 29 | 10 | 638 | 130 |
| stroma | 5 | 71 | 1 | 24 | 47 | 611 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7177    0.8900    0.7946       100
      cancer     0.8603    0.8111    0.8350      1329
       empty     0.9794    0.9981    0.9887      1621
      normal     0.9504    0.8698    0.9083      1014
       other     0.6802    0.7258    0.7023       879
      stroma     0.7571    0.8050    0.7803       759

    accuracy                         0.8622      5702
   macro avg     0.8242    0.8500    0.8349      5702
weighted avg     0.8662    0.8622    0.8633      5702

```