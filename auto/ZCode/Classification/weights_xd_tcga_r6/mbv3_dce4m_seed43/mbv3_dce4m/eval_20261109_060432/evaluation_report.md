# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_dce4m_seed43/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8173 |
| Accuracy | 0.8593 |
| Macro-AUROC (OvR) | 0.9759 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8631 |
| Balanced Acc | 0.8420 |
| Macro-Sensitivity | 0.8420 |
| Macro-Specificity | 0.9725 |
| Macro-Precision | 0.8046 |
| AUC-PR (Macro) | 0.8789 |
| Cohen's Kappa | 0.8234 |
| MCC | 0.8248 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 8756.3 (0.11 ms/张) |
| End-to-End FPS | 562.1 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8400 | 0.9902 | 0.6043 | 0.7029 | 0.9971 | 0.9934 | 0.8469 | 100 |
| cancer | 0.7780 | 0.9838 | 0.9357 | 0.8496 | 0.9358 | 0.9759 | 0.9367 | 1329 |
| empty | 0.9944 | 0.9971 | 0.9926 | 0.9935 | 0.9978 | 0.9996 | 0.9994 | 1621 |
| normal | 0.8974 | 0.9915 | 0.9579 | 0.9267 | 0.9781 | 0.9888 | 0.9772 | 1014 |
| other | 0.7452 | 0.9374 | 0.6844 | 0.7135 | 0.9528 | 0.9392 | 0.8211 | 879 |
| stroma | 0.7971 | 0.9349 | 0.6526 | 0.7177 | 0.9677 | 0.9584 | 0.6923 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=84 FP=55 TN=5547 FN=16
- **cancer**: TP=1034 FP=71 TN=4302 FN=295
- **empty**: TP=1612 FP=12 TN=4069 FN=9
- **normal**: TP=910 FP=40 TN=4648 FN=104
- **other**: TP=655 FP=302 TN=4521 FN=224
- **stroma**: TP=605 FP=322 TN=4621 FN=154

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 84 | 1 | 2 | 3 | 8 | 2 |
| cancer | 38 | 1034 | 0 | 18 | 136 | 103 |
| empty | 0 | 1 | 1612 | 0 | 6 | 2 |
| normal | 6 | 21 | 0 | 910 | 32 | 45 |
| other | 9 | 30 | 10 | 5 | 655 | 170 |
| stroma | 2 | 18 | 0 | 14 | 120 | 605 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6043    0.8400    0.7029       100
      cancer     0.9357    0.7780    0.8496      1329
       empty     0.9926    0.9944    0.9935      1621
      normal     0.9579    0.8974    0.9267      1014
       other     0.6844    0.7452    0.7135       879
      stroma     0.6526    0.7971    0.7177       759

    accuracy                         0.8593      5702
   macro avg     0.8046    0.8420    0.8173      5702
weighted avg     0.8736    0.8593    0.8631      5702

```