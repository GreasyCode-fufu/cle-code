# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_egb_scg_seed43/convnext_egb_scg/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8112 |
| Accuracy | 0.8399 |
| Macro-AUROC (OvR) | 0.9688 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8313 |
| Balanced Acc | 0.8083 |
| Macro-Sensitivity | 0.8083 |
| Macro-Specificity | 0.9660 |
| Macro-Precision | 0.8279 |
| AUC-PR (Macro) | 0.8879 |
| Cohen's Kappa | 0.7928 |
| MCC | 0.7965 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.29M |
| Pure-Inference FPS | 739.3 (1.35 ms/张) |
| End-to-End FPS | 351.4 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8995 | 0.9973 | 0.8647 | 0.8818 | 0.9981 | 0.9981 | 0.9525 | 199 |
| cancer | 0.9539 | 0.9118 | 0.7816 | 0.8592 | 0.9835 | 0.9838 | 0.9598 | 2645 |
| empty | 0.9957 | 0.9649 | 0.9260 | 0.9596 | 0.9980 | 0.9855 | 0.9654 | 3256 |
| normal | 0.7648 | 0.9795 | 0.8794 | 0.8181 | 0.9553 | 0.9785 | 0.9172 | 1735 |
| other | 0.5084 | 0.9712 | 0.7726 | 0.6133 | 0.9112 | 0.9024 | 0.7272 | 1717 |
| stroma | 0.7274 | 0.9715 | 0.7432 | 0.7352 | 0.9692 | 0.9644 | 0.8051 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=179 FP=28 TN=10407 FN=20
- **cancer**: TP=2523 FP=705 TN=7284 FN=122
- **empty**: TP=3242 FP=259 TN=7119 FN=14
- **normal**: TP=1327 FP=182 TN=8717 FN=408
- **other**: TP=873 FP=257 TN=8660 FN=844
- **stroma**: TP=787 FP=272 TN=9280 FN=295

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 179 | 5 | 1 | 10 | 2 | 2 |
| cancer | 0 | 2523 | 2 | 78 | 31 | 11 |
| empty | 1 | 3 | 3242 | 0 | 10 | 0 |
| normal | 0 | 201 | 1 | 1327 | 133 | 73 |
| other | 23 | 348 | 255 | 32 | 873 | 186 |
| stroma | 4 | 148 | 0 | 62 | 81 | 787 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8647    0.8995    0.8818       199
      cancer     0.7816    0.9539    0.8592      2645
       empty     0.9260    0.9957    0.9596      3256
      normal     0.8794    0.7648    0.8181      1735
       other     0.7726    0.5084    0.6133      1717
      stroma     0.7432    0.7274    0.7352      1082

    accuracy                         0.8399     10634
   macro avg     0.8279    0.8083    0.8112     10634
weighted avg     0.8380    0.8399    0.8313     10634

```