# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce4m_seed44/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8406 |
| Accuracy | 0.8732 |
| Macro-AUROC (OvR) | 0.9757 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8756 |
| Balanced Acc | 0.8384 |
| Macro-Sensitivity | 0.8384 |
| Macro-Specificity | 0.9748 |
| Macro-Precision | 0.8468 |
| AUC-PR (Macro) | 0.8837 |
| Cohen's Kappa | 0.8401 |
| MCC | 0.8408 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 8793.4 (0.11 ms/张) |
| End-to-End FPS | 565.1 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7600 | 0.9968 | 0.8085 | 0.7835 | 0.9957 | 0.9931 | 0.8377 | 100 |
| cancer | 0.8653 | 0.9673 | 0.8894 | 0.8772 | 0.9594 | 0.9767 | 0.9470 | 1329 |
| empty | 0.9951 | 0.9971 | 0.9926 | 0.9938 | 0.9980 | 0.9999 | 0.9998 | 1621 |
| normal | 0.8639 | 0.9921 | 0.9595 | 0.9092 | 0.9712 | 0.9890 | 0.9696 | 1014 |
| other | 0.7531 | 0.9631 | 0.7881 | 0.7702 | 0.9554 | 0.9458 | 0.8597 | 879 |
| stroma | 0.7931 | 0.9322 | 0.6425 | 0.7099 | 0.9671 | 0.9495 | 0.6886 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=76 FP=18 TN=5584 FN=24
- **cancer**: TP=1150 FP=143 TN=4230 FN=179
- **empty**: TP=1613 FP=12 TN=4069 FN=8
- **normal**: TP=876 FP=37 TN=4651 FN=138
- **other**: TP=662 FP=178 TN=4645 FN=217
- **stroma**: TP=602 FP=335 TN=4608 FN=157

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 76 | 4 | 1 | 3 | 9 | 7 |
| cancer | 9 | 1150 | 1 | 13 | 50 | 106 |
| empty | 0 | 0 | 1613 | 0 | 6 | 2 |
| normal | 4 | 37 | 0 | 876 | 35 | 62 |
| other | 2 | 41 | 9 | 7 | 662 | 158 |
| stroma | 3 | 61 | 1 | 14 | 78 | 602 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8085    0.7600    0.7835       100
      cancer     0.8894    0.8653    0.8772      1329
       empty     0.9926    0.9951    0.9938      1621
      normal     0.9595    0.8639    0.9092      1014
       other     0.7881    0.7531    0.7702       879
      stroma     0.6425    0.7931    0.7099       759

    accuracy                         0.8732      5702
   macro avg     0.8468    0.8384    0.8406      5702
weighted avg     0.8813    0.8732    0.8756      5702

```