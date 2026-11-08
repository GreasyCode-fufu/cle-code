# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce4m_seed43/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8389 |
| Accuracy | 0.8750 |
| Macro-AUROC (OvR) | 0.9770 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8762 |
| Balanced Acc | 0.8389 |
| Macro-Sensitivity | 0.8389 |
| Macro-Specificity | 0.9750 |
| Macro-Precision | 0.8451 |
| AUC-PR (Macro) | 0.8799 |
| Cohen's Kappa | 0.8421 |
| MCC | 0.8432 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 8744.0 (0.11 ms/张) |
| End-to-End FPS | 560.1 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7600 | 0.9959 | 0.7677 | 0.7638 | 0.9957 | 0.9898 | 0.8099 | 100 |
| cancer | 0.8819 | 0.9575 | 0.8630 | 0.8723 | 0.9639 | 0.9785 | 0.9463 | 1329 |
| empty | 0.9926 | 0.9963 | 0.9908 | 0.9917 | 0.9971 | 0.9999 | 0.9998 | 1621 |
| normal | 0.8945 | 0.9902 | 0.9517 | 0.9222 | 0.9775 | 0.9937 | 0.9795 | 1014 |
| other | 0.6928 | 0.9776 | 0.8494 | 0.7632 | 0.9458 | 0.9462 | 0.8549 | 879 |
| stroma | 0.8116 | 0.9322 | 0.6477 | 0.7205 | 0.9699 | 0.9539 | 0.6889 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=76 FP=23 TN=5579 FN=24
- **cancer**: TP=1172 FP=186 TN=4187 FN=157
- **empty**: TP=1609 FP=15 TN=4066 FN=12
- **normal**: TP=907 FP=46 TN=4642 FN=107
- **other**: TP=609 FP=108 TN=4715 FN=270
- **stroma**: TP=616 FP=335 TN=4608 FN=143

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 76 | 4 | 2 | 5 | 7 | 6 |
| cancer | 20 | 1172 | 1 | 17 | 31 | 88 |
| empty | 0 | 0 | 1609 | 1 | 6 | 5 |
| normal | 1 | 24 | 0 | 907 | 29 | 53 |
| other | 0 | 67 | 11 | 9 | 609 | 183 |
| stroma | 2 | 91 | 1 | 14 | 35 | 616 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7677    0.7600    0.7638       100
      cancer     0.8630    0.8819    0.8723      1329
       empty     0.9908    0.9926    0.9917      1621
      normal     0.9517    0.8945    0.9222      1014
       other     0.8494    0.6928    0.7632       879
      stroma     0.6477    0.8116    0.7205       759

    accuracy                         0.8750      5702
   macro avg     0.8451    0.8389    0.8389      5702
weighted avg     0.8827    0.8750    0.8762      5702

```