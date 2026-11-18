# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_abl/convnext_v2c_seed43/convnext_v2c/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8152 |
| Accuracy | 0.8518 |
| Macro-AUROC (OvR) | 0.9669 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8545 |
| Balanced Acc | 0.8394 |
| Macro-Sensitivity | 0.8394 |
| Macro-Specificity | 0.9709 |
| Macro-Precision | 0.8038 |
| AUC-PR (Macro) | 0.8569 |
| Cohen's Kappa | 0.8139 |
| MCC | 0.8159 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.26M |
| Pure-Inference FPS | 601.9 (1.66 ms/张) |
| End-to-End FPS | 314.8 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8600 | 0.9914 | 0.6418 | 0.7350 | 0.9975 | 0.9941 | 0.8704 | 100 |
| cancer | 0.7344 | 0.9783 | 0.9113 | 0.8133 | 0.9238 | 0.9389 | 0.8925 | 1329 |
| empty | 0.9963 | 0.9980 | 0.9951 | 0.9957 | 0.9985 | 1.0000 | 0.9999 | 1621 |
| normal | 0.9152 | 0.9748 | 0.8872 | 0.9010 | 0.9815 | 0.9882 | 0.9540 | 1014 |
| other | 0.7509 | 0.9641 | 0.7923 | 0.7710 | 0.9550 | 0.9478 | 0.8529 | 879 |
| stroma | 0.7800 | 0.9185 | 0.5950 | 0.6750 | 0.9645 | 0.9325 | 0.5714 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=86 FP=48 TN=5554 FN=14
- **cancer**: TP=976 FP=95 TN=4278 FN=353
- **empty**: TP=1615 FP=8 TN=4073 FN=6
- **normal**: TP=928 FP=118 TN=4570 FN=86
- **other**: TP=660 FP=173 TN=4650 FN=219
- **stroma**: TP=592 FP=403 TN=4540 FN=167

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 86 | 0 | 1 | 4 | 5 | 4 |
| cancer | 38 | 976 | 1 | 47 | 61 | 206 |
| empty | 1 | 0 | 1615 | 1 | 3 | 1 |
| normal | 3 | 25 | 0 | 928 | 23 | 35 |
| other | 2 | 38 | 6 | 16 | 660 | 157 |
| stroma | 4 | 32 | 0 | 50 | 81 | 592 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6418    0.8600    0.7350       100
      cancer     0.9113    0.7344    0.8133      1329
       empty     0.9951    0.9963    0.9957      1621
      normal     0.8872    0.9152    0.9010      1014
       other     0.7923    0.7509    0.7710       879
      stroma     0.5950    0.7800    0.6750       759

    accuracy                         0.8518      5702
   macro avg     0.8038    0.8394    0.8152      5702
weighted avg     0.8657    0.8518    0.8545      5702

```