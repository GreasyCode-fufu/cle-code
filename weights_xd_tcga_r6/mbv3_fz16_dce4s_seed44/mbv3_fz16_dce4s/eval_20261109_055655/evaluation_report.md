# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce4s_seed44/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7915 |
| Accuracy | 0.8349 |
| Macro-AUROC (OvR) | 0.9517 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8309 |
| Balanced Acc | 0.7904 |
| Macro-Sensitivity | 0.7904 |
| Macro-Specificity | 0.9657 |
| Macro-Precision | 0.7979 |
| AUC-PR (Macro) | 0.8477 |
| Cohen's Kappa | 0.7870 |
| MCC | 0.7880 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9600.7 (0.10 ms/张) |
| End-to-End FPS | 548.7 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8593 | 0.9953 | 0.7773 | 0.8162 | 0.9973 | 0.9961 | 0.9024 | 199 |
| cancer | 0.9233 | 0.9543 | 0.8700 | 0.8958 | 0.9741 | 0.9871 | 0.9647 | 2645 |
| empty | 0.9957 | 0.9602 | 0.9169 | 0.9547 | 0.9980 | 0.9882 | 0.9655 | 3256 |
| normal | 0.7516 | 0.9836 | 0.8993 | 0.8188 | 0.9531 | 0.9448 | 0.8796 | 1735 |
| other | 0.6407 | 0.9311 | 0.6418 | 0.6412 | 0.9308 | 0.8692 | 0.6770 | 1717 |
| stroma | 0.5721 | 0.9698 | 0.6825 | 0.6224 | 0.9524 | 0.9252 | 0.6972 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=171 FP=49 TN=10386 FN=28
- **cancer**: TP=2442 FP=365 TN=7624 FN=203
- **empty**: TP=3242 FP=294 TN=7084 FN=14
- **normal**: TP=1304 FP=146 TN=8753 FN=431
- **other**: TP=1100 FP=614 TN=8303 FN=617
- **stroma**: TP=619 FP=288 TN=9264 FN=463

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 171 | 5 | 2 | 6 | 12 | 3 |
| cancer | 6 | 2442 | 1 | 39 | 107 | 50 |
| empty | 0 | 0 | 3242 | 0 | 14 | 0 |
| normal | 1 | 143 | 3 | 1304 | 187 | 97 |
| other | 39 | 138 | 288 | 14 | 1100 | 138 |
| stroma | 3 | 79 | 0 | 87 | 294 | 619 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7773    0.8593    0.8162       199
      cancer     0.8700    0.9233    0.8958      2645
       empty     0.9169    0.9957    0.9547      3256
      normal     0.8993    0.7516    0.8188      1735
       other     0.6418    0.6407    0.6412      1717
      stroma     0.6825    0.5721    0.6224      1082

    accuracy                         0.8349     10634
   macro avg     0.7979    0.7904    0.7915     10634
weighted avg     0.8315    0.8349    0.8309     10634

```