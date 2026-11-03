# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/repvit_m1_seed43/repvit_m1/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7618 |
| Accuracy | 0.8143 |
| Macro-AUROC (OvR) | 0.9625 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8157 |
| Balanced Acc | 0.7720 |
| Macro-Sensitivity | 0.7720 |
| Macro-Specificity | 0.9630 |
| Macro-Precision | 0.7626 |
| AUC-PR (Macro) | 0.8258 |
| Cohen's Kappa | 0.7659 |
| MCC | 0.7678 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3155.3 (0.32 ms/张) |
| End-to-End FPS | 542.9 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7000 | 0.9918 | 0.6034 | 0.6481 | 0.9946 | 0.9766 | 0.7348 | 100 |
| cancer | 0.8533 | 0.9719 | 0.9021 | 0.8770 | 0.9561 | 0.9781 | 0.9437 | 1329 |
| empty | 0.9963 | 0.9863 | 0.9665 | 0.9812 | 0.9985 | 0.9993 | 0.9974 | 1621 |
| normal | 0.6292 | 0.9797 | 0.8704 | 0.7304 | 0.9243 | 0.9660 | 0.8682 | 1014 |
| other | 0.6906 | 0.9175 | 0.6040 | 0.6444 | 0.9421 | 0.9118 | 0.7024 | 879 |
| stroma | 0.7628 | 0.9310 | 0.6293 | 0.6897 | 0.9624 | 0.9432 | 0.7085 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=70 FP=46 TN=5556 FN=30
- **cancer**: TP=1134 FP=123 TN=4250 FN=195
- **empty**: TP=1615 FP=56 TN=4025 FN=6
- **normal**: TP=638 FP=95 TN=4593 FN=376
- **other**: TP=607 FP=398 TN=4425 FN=272
- **stroma**: TP=579 FP=341 TN=4602 FN=180

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 70 | 8 | 4 | 8 | 7 | 3 |
| cancer | 6 | 1134 | 1 | 57 | 31 | 100 |
| empty | 0 | 1 | 1615 | 0 | 5 | 0 |
| normal | 3 | 34 | 0 | 638 | 249 | 90 |
| other | 20 | 44 | 50 | 10 | 607 | 148 |
| stroma | 17 | 36 | 1 | 20 | 106 | 579 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6034    0.7000    0.6481       100
      cancer     0.9021    0.8533    0.8770      1329
       empty     0.9665    0.9963    0.9812      1621
      normal     0.8704    0.6292    0.7304      1014
       other     0.6040    0.6906    0.6444       879
      stroma     0.6293    0.7628    0.6897       759

    accuracy                         0.8143      5702
   macro avg     0.7626    0.7720    0.7618      5702
weighted avg     0.8273    0.8143    0.8157      5702

```