# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/fast_t0_seed43/fast_t0/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8421 |
| Accuracy | 0.8758 |
| Macro-AUROC (OvR) | 0.9781 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8787 |
| Balanced Acc | 0.8616 |
| Macro-Sensitivity | 0.8616 |
| Macro-Specificity | 0.9756 |
| Macro-Precision | 0.8307 |
| AUC-PR (Macro) | 0.9015 |
| Cohen's Kappa | 0.8440 |
| MCC | 0.8451 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 8583.3 (0.12 ms/张) |
| End-to-End FPS | 606.8 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8500 | 0.9927 | 0.6746 | 0.7522 | 0.9973 | 0.9964 | 0.8857 | 100 |
| cancer | 0.7953 | 0.9760 | 0.9096 | 0.8487 | 0.9401 | 0.9712 | 0.9358 | 1329 |
| empty | 0.9889 | 0.9995 | 0.9988 | 0.9938 | 0.9956 | 0.9999 | 0.9999 | 1621 |
| normal | 0.8915 | 0.9910 | 0.9556 | 0.9224 | 0.9769 | 0.9920 | 0.9728 | 1014 |
| other | 0.8123 | 0.9581 | 0.7795 | 0.7955 | 0.9655 | 0.9498 | 0.8779 | 879 |
| stroma | 0.8314 | 0.9361 | 0.6663 | 0.7397 | 0.9731 | 0.9592 | 0.7372 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=85 FP=41 TN=5561 FN=15
- **cancer**: TP=1057 FP=105 TN=4268 FN=272
- **empty**: TP=1603 FP=2 TN=4079 FN=18
- **normal**: TP=904 FP=42 TN=4646 FN=110
- **other**: TP=714 FP=202 TN=4621 FN=165
- **stroma**: TP=631 FP=316 TN=4627 FN=128

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 85 | 1 | 0 | 5 | 6 | 3 |
| cancer | 22 | 1057 | 1 | 17 | 74 | 158 |
| empty | 1 | 1 | 1603 | 0 | 14 | 2 |
| normal | 7 | 41 | 0 | 904 | 28 | 34 |
| other | 4 | 38 | 1 | 3 | 714 | 119 |
| stroma | 7 | 24 | 0 | 17 | 80 | 631 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6746    0.8500    0.7522       100
      cancer     0.9096    0.7953    0.8487      1329
       empty     0.9988    0.9889    0.9938      1621
      normal     0.9556    0.8915    0.9224      1014
       other     0.7795    0.8123    0.7955       879
      stroma     0.6663    0.8314    0.7397       759

    accuracy                         0.8758      5702
   macro avg     0.8307    0.8616    0.8421      5702
weighted avg     0.8866    0.8758    0.8787      5702

```