# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce4s_seed43/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8391 |
| Accuracy | 0.8753 |
| Macro-AUROC (OvR) | 0.9770 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8765 |
| Balanced Acc | 0.8388 |
| Macro-Sensitivity | 0.8388 |
| Macro-Specificity | 0.9750 |
| Macro-Precision | 0.8455 |
| AUC-PR (Macro) | 0.8799 |
| Cohen's Kappa | 0.8425 |
| MCC | 0.8436 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9095.9 (0.11 ms/张) |
| End-to-End FPS | 561.5 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7600 | 0.9959 | 0.7677 | 0.7638 | 0.9957 | 0.9899 | 0.8105 | 100 |
| cancer | 0.8834 | 0.9568 | 0.8613 | 0.8722 | 0.9643 | 0.9785 | 0.9462 | 1329 |
| empty | 0.9951 | 0.9963 | 0.9908 | 0.9929 | 0.9980 | 0.9999 | 0.9999 | 1621 |
| normal | 0.8915 | 0.9904 | 0.9526 | 0.9210 | 0.9769 | 0.9937 | 0.9794 | 1014 |
| other | 0.6928 | 0.9780 | 0.8517 | 0.7641 | 0.9459 | 0.9464 | 0.8554 | 879 |
| stroma | 0.8103 | 0.9326 | 0.6487 | 0.7206 | 0.9697 | 0.9537 | 0.6882 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=76 FP=23 TN=5579 FN=24
- **cancer**: TP=1174 FP=189 TN=4184 FN=155
- **empty**: TP=1613 FP=15 TN=4066 FN=8
- **normal**: TP=904 FP=45 TN=4643 FN=110
- **other**: TP=609 FP=106 TN=4717 FN=270
- **stroma**: TP=615 FP=333 TN=4610 FN=144

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 76 | 4 | 2 | 5 | 7 | 6 |
| cancer | 20 | 1174 | 1 | 16 | 30 | 88 |
| empty | 0 | 0 | 1613 | 1 | 6 | 1 |
| normal | 1 | 25 | 0 | 904 | 29 | 55 |
| other | 0 | 67 | 11 | 9 | 609 | 183 |
| stroma | 2 | 93 | 1 | 14 | 34 | 615 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7677    0.7600    0.7638       100
      cancer     0.8613    0.8834    0.8722      1329
       empty     0.9908    0.9951    0.9929      1621
      normal     0.9526    0.8915    0.9210      1014
       other     0.8517    0.6928    0.7641       879
      stroma     0.6487    0.8103    0.7206       759

    accuracy                         0.8753      5702
   macro avg     0.8455    0.8388    0.8391      5702
weighted avg     0.8829    0.8753    0.8765      5702

```