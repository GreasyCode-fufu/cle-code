# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_dce4m_seed43/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8166 |
| Accuracy | 0.8541 |
| Macro-AUROC (OvR) | 0.9585 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8525 |
| Balanced Acc | 0.8295 |
| Macro-Sensitivity | 0.8295 |
| Macro-Specificity | 0.9701 |
| Macro-Precision | 0.8117 |
| AUC-PR (Macro) | 0.8648 |
| Cohen's Kappa | 0.8126 |
| MCC | 0.8136 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9259.4 (0.11 ms/张) |
| End-to-End FPS | 571.6 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.9246 | 0.9936 | 0.7331 | 0.8178 | 0.9986 | 0.9980 | 0.9217 | 199 |
| cancer | 0.9225 | 0.9750 | 0.9242 | 0.9234 | 0.9744 | 0.9912 | 0.9763 | 2645 |
| empty | 0.9966 | 0.9611 | 0.9187 | 0.9561 | 0.9985 | 0.9751 | 0.8813 | 3256 |
| normal | 0.7331 | 0.9881 | 0.9231 | 0.8172 | 0.9500 | 0.9621 | 0.8927 | 1735 |
| other | 0.6721 | 0.9406 | 0.6853 | 0.6786 | 0.9371 | 0.8713 | 0.7358 | 1717 |
| stroma | 0.7283 | 0.9622 | 0.6858 | 0.7064 | 0.9690 | 0.9533 | 0.7810 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=184 FP=67 TN=10368 FN=15
- **cancer**: TP=2440 FP=200 TN=7789 FN=205
- **empty**: TP=3245 FP=287 TN=7091 FN=11
- **normal**: TP=1272 FP=106 TN=8793 FN=463
- **other**: TP=1154 FP=530 TN=8387 FN=563
- **stroma**: TP=788 FP=361 TN=9191 FN=294

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 184 | 3 | 2 | 0 | 7 | 3 |
| cancer | 8 | 2440 | 3 | 39 | 92 | 63 |
| empty | 0 | 3 | 3245 | 0 | 8 | 0 |
| normal | 2 | 53 | 3 | 1272 | 230 | 175 |
| other | 53 | 97 | 279 | 14 | 1154 | 120 |
| stroma | 4 | 44 | 0 | 53 | 193 | 788 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7331    0.9246    0.8178       199
      cancer     0.9242    0.9225    0.9234      2645
       empty     0.9187    0.9966    0.9561      3256
      normal     0.9231    0.7331    0.8172      1735
       other     0.6853    0.6721    0.6786      1717
      stroma     0.6858    0.7283    0.7064      1082

    accuracy                         0.8541     10634
   macro avg     0.8117    0.8295    0.8166     10634
weighted avg     0.8559    0.8541    0.8525     10634

```