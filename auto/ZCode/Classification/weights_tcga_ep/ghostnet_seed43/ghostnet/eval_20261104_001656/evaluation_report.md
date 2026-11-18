# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/ghostnet_seed43/ghostnet/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7986 |
| Accuracy | 0.8376 |
| Macro-AUROC (OvR) | 0.9428 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8367 |
| Balanced Acc | 0.8022 |
| Macro-Sensitivity | 0.8022 |
| Macro-Specificity | 0.9667 |
| Macro-Precision | 0.8016 |
| AUC-PR (Macro) | 0.8312 |
| Cohen's Kappa | 0.7912 |
| MCC | 0.7924 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.59M |
| FLOPs | 309.17M |
| Params | 3.91M |
| Pure-Inference FPS | 4877.5 (0.21 ms/张) |
| End-to-End FPS | 569.3 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8693 | 0.9948 | 0.7621 | 0.8122 | 0.9975 | 0.9942 | 0.8937 | 199 |
| cancer | 0.9013 | 0.9716 | 0.9131 | 0.9072 | 0.9675 | 0.9882 | 0.9672 | 2645 |
| empty | 0.9923 | 0.9606 | 0.9174 | 0.9534 | 0.9965 | 0.9721 | 0.8732 | 3256 |
| normal | 0.7061 | 0.9871 | 0.9142 | 0.7967 | 0.9451 | 0.9290 | 0.8541 | 1735 |
| other | 0.6919 | 0.9219 | 0.6306 | 0.6598 | 0.9395 | 0.8559 | 0.6782 | 1717 |
| stroma | 0.6525 | 0.9640 | 0.6724 | 0.6623 | 0.9608 | 0.9172 | 0.7206 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=173 FP=54 TN=10381 FN=26
- **cancer**: TP=2384 FP=227 TN=7762 FN=261
- **empty**: TP=3231 FP=291 TN=7087 FN=25
- **normal**: TP=1225 FP=115 TN=8784 FN=510
- **other**: TP=1188 FP=696 TN=8221 FN=529
- **stroma**: TP=706 FP=344 TN=9208 FN=376

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 173 | 6 | 2 | 6 | 11 | 1 |
| cancer | 7 | 2384 | 2 | 34 | 144 | 74 |
| empty | 0 | 5 | 3231 | 0 | 20 | 0 |
| normal | 1 | 82 | 0 | 1225 | 251 | 176 |
| other | 43 | 86 | 287 | 20 | 1188 | 93 |
| stroma | 3 | 48 | 0 | 55 | 270 | 706 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7621    0.8693    0.8122       199
      cancer     0.9131    0.9013    0.9072      2645
       empty     0.9174    0.9923    0.9534      3256
      normal     0.9142    0.7061    0.7967      1735
       other     0.6306    0.6919    0.6598      1717
      stroma     0.6724    0.6525    0.6623      1082

    accuracy                         0.8376     10634
   macro avg     0.8016    0.8022    0.7986     10634
weighted avg     0.8416    0.8376    0.8367     10634

```