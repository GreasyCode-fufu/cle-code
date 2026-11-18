# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/pvt_tiny_seed42/pvt_tiny/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7800 |
| Accuracy | 0.8130 |
| Macro-AUROC (OvR) | 0.9497 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8121 |
| Balanced Acc | 0.7829 |
| Macro-Sensitivity | 0.7829 |
| Macro-Specificity | 0.9617 |
| Macro-Precision | 0.7796 |
| AUC-PR (Macro) | 0.8270 |
| Cohen's Kappa | 0.7602 |
| MCC | 0.7609 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 1317.2 (0.76 ms/张) |
| End-to-End FPS | 436.6 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8392 | 0.9962 | 0.8068 | 0.8227 | 0.9969 | 0.9973 | 0.8960 | 199 |
| cancer | 0.8079 | 0.9611 | 0.8730 | 0.8392 | 0.9379 | 0.9361 | 0.9012 | 2645 |
| empty | 0.9936 | 0.9620 | 0.9203 | 0.9555 | 0.9971 | 0.9737 | 0.8737 | 3256 |
| normal | 0.7516 | 0.9722 | 0.8407 | 0.7937 | 0.9525 | 0.9723 | 0.8874 | 1735 |
| other | 0.6139 | 0.9277 | 0.6204 | 0.6171 | 0.9258 | 0.8721 | 0.6757 | 1717 |
| stroma | 0.6913 | 0.9512 | 0.6161 | 0.6516 | 0.9645 | 0.9468 | 0.7280 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=167 FP=40 TN=10395 FN=32
- **cancer**: TP=2137 FP=311 TN=7678 FN=508
- **empty**: TP=3235 FP=280 TN=7098 FN=21
- **normal**: TP=1304 FP=247 TN=8652 FN=431
- **other**: TP=1054 FP=645 TN=8272 FN=663
- **stroma**: TP=748 FP=466 TN=9086 FN=334

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 167 | 9 | 2 | 5 | 14 | 2 |
| cancer | 1 | 2137 | 0 | 124 | 300 | 83 |
| empty | 0 | 0 | 3235 | 0 | 20 | 1 |
| normal | 2 | 133 | 1 | 1304 | 108 | 187 |
| other | 35 | 118 | 277 | 40 | 1054 | 193 |
| stroma | 2 | 51 | 0 | 78 | 203 | 748 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8068    0.8392    0.8227       199
      cancer     0.8730    0.8079    0.8392      2645
       empty     0.9203    0.9936    0.9555      3256
      normal     0.8407    0.7516    0.7937      1735
       other     0.6204    0.6139    0.6171      1717
      stroma     0.6161    0.6913    0.6516      1082

    accuracy                         0.8130     10634
   macro avg     0.7796    0.7829    0.7800     10634
weighted avg     0.8141    0.8130    0.8121     10634

```