# convnext_egb_scg_hsc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_egb_scg_hsc_seed44/convnext_egb_scg_hsc/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8068 |
| Accuracy | 0.8412 |
| Macro-AUROC (OvR) | 0.9634 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8396 |
| Balanced Acc | 0.8080 |
| Macro-Sensitivity | 0.8080 |
| Macro-Specificity | 0.9671 |
| Macro-Precision | 0.8116 |
| AUC-PR (Macro) | 0.8713 |
| Cohen's Kappa | 0.7955 |
| MCC | 0.7966 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.58M |
| Pure-Inference FPS | 1055.7 (0.95 ms/张) |
| End-to-End FPS | 411.3 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8794 | 0.9952 | 0.7778 | 0.8255 | 0.9977 | 0.9964 | 0.8899 | 199 |
| cancer | 0.8941 | 0.9597 | 0.8802 | 0.8871 | 0.9648 | 0.9837 | 0.9609 | 2645 |
| empty | 0.9975 | 0.9620 | 0.9206 | 0.9575 | 0.9989 | 0.9807 | 0.9461 | 3256 |
| normal | 0.7337 | 0.9882 | 0.9238 | 0.8179 | 0.9501 | 0.9615 | 0.8939 | 1735 |
| other | 0.6779 | 0.9251 | 0.6354 | 0.6560 | 0.9372 | 0.8937 | 0.7414 | 1717 |
| stroma | 0.6654 | 0.9724 | 0.7317 | 0.6970 | 0.9625 | 0.9643 | 0.7955 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=175 FP=50 TN=10385 FN=24
- **cancer**: TP=2365 FP=322 TN=7667 FN=280
- **empty**: TP=3248 FP=280 TN=7098 FN=8
- **normal**: TP=1273 FP=105 TN=8794 FN=462
- **other**: TP=1164 FP=668 TN=8249 FN=553
- **stroma**: TP=720 FP=264 TN=9288 FN=362

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 175 | 5 | 2 | 1 | 15 | 1 |
| cancer | 4 | 2365 | 3 | 41 | 194 | 38 |
| empty | 0 | 0 | 3248 | 0 | 8 | 0 |
| normal | 4 | 146 | 0 | 1273 | 212 | 100 |
| other | 35 | 108 | 275 | 10 | 1164 | 125 |
| stroma | 7 | 63 | 0 | 53 | 239 | 720 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7778    0.8794    0.8255       199
      cancer     0.8802    0.8941    0.8871      2645
       empty     0.9206    0.9975    0.9575      3256
      normal     0.9238    0.7337    0.8179      1735
       other     0.6354    0.6779    0.6560      1717
      stroma     0.7317    0.6654    0.6970      1082

    accuracy                         0.8412     10634
   macro avg     0.8116    0.8080    0.8068     10634
weighted avg     0.8431    0.8412    0.8396     10634

```