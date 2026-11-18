# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mobilenet_v3_large_seed44/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8273 |
| Accuracy | 0.8688 |
| Macro-AUROC (OvR) | 0.9782 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8708 |
| Balanced Acc | 0.8471 |
| Macro-Sensitivity | 0.8471 |
| Macro-Specificity | 0.9740 |
| Macro-Precision | 0.8140 |
| AUC-PR (Macro) | 0.8869 |
| Cohen's Kappa | 0.8348 |
| MCC | 0.8352 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.93M |
| Params | 4.21M |
| Pure-Inference FPS | 6431.8 (0.16 ms/张) |
| End-to-End FPS | 226.5 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8400 | 0.9913 | 0.6316 | 0.7210 | 0.9971 | 0.9948 | 0.8787 | 100 |
| cancer | 0.8179 | 0.9716 | 0.8976 | 0.8559 | 0.9461 | 0.9761 | 0.9386 | 1329 |
| empty | 0.9951 | 0.9963 | 0.9908 | 0.9929 | 0.9980 | 0.9998 | 0.9996 | 1621 |
| normal | 0.9122 | 0.9883 | 0.9439 | 0.9278 | 0.9812 | 0.9942 | 0.9789 | 1014 |
| other | 0.7793 | 0.9560 | 0.7637 | 0.7714 | 0.9596 | 0.9583 | 0.8494 | 879 |
| stroma | 0.7378 | 0.9407 | 0.6565 | 0.6948 | 0.9590 | 0.9460 | 0.6761 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=84 FP=49 TN=5553 FN=16
- **cancer**: TP=1087 FP=124 TN=4249 FN=242
- **empty**: TP=1613 FP=15 TN=4066 FN=8
- **normal**: TP=925 FP=55 TN=4633 FN=89
- **other**: TP=685 FP=212 TN=4611 FN=194
- **stroma**: TP=560 FP=293 TN=4650 FN=199

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 84 | 2 | 2 | 4 | 2 | 6 |
| cancer | 19 | 1087 | 0 | 31 | 60 | 132 |
| empty | 1 | 1 | 1613 | 0 | 5 | 1 |
| normal | 7 | 26 | 0 | 925 | 27 | 29 |
| other | 8 | 43 | 13 | 5 | 685 | 125 |
| stroma | 14 | 52 | 0 | 15 | 118 | 560 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6316    0.8400    0.7210       100
      cancer     0.8976    0.8179    0.8559      1329
       empty     0.9908    0.9951    0.9929      1621
      normal     0.9439    0.9122    0.9278      1014
       other     0.7637    0.7793    0.7714       879
      stroma     0.6565    0.7378    0.6948       759

    accuracy                         0.8688      5702
   macro avg     0.8140    0.8471    0.8273      5702
weighted avg     0.8749    0.8688    0.8708      5702

```