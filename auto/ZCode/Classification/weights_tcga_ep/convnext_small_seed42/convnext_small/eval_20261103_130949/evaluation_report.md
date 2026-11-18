# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_small_seed42/convnext_small/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8570 |
| Accuracy | 0.8869 |
| Macro-AUROC (OvR) | 0.9820 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8860 |
| Balanced Acc | 0.8527 |
| Macro-Sensitivity | 0.8527 |
| Macro-Specificity | 0.9774 |
| Macro-Precision | 0.8646 |
| AUC-PR (Macro) | 0.9155 |
| Cohen's Kappa | 0.8571 |
| MCC | 0.8578 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.42M |
| Pure-Inference FPS | 406.5 (2.46 ms/张) |
| End-to-End FPS | 251.7 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7800 | 0.9979 | 0.8667 | 0.8211 | 0.9961 | 0.9946 | 0.8983 | 100 |
| cancer | 0.8525 | 0.9783 | 0.9226 | 0.8862 | 0.9562 | 0.9819 | 0.9498 | 1329 |
| empty | 0.9957 | 0.9985 | 0.9963 | 0.9960 | 0.9983 | 1.0000 | 0.9999 | 1621 |
| normal | 0.9813 | 0.9597 | 0.8404 | 0.9054 | 0.9958 | 0.9952 | 0.9818 | 1014 |
| other | 0.7793 | 0.9677 | 0.8145 | 0.7965 | 0.9601 | 0.9599 | 0.8822 | 879 |
| stroma | 0.7273 | 0.9622 | 0.7470 | 0.7370 | 0.9583 | 0.9604 | 0.7810 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=78 FP=12 TN=5590 FN=22
- **cancer**: TP=1133 FP=95 TN=4278 FN=196
- **empty**: TP=1614 FP=6 TN=4075 FN=7
- **normal**: TP=995 FP=189 TN=4499 FN=19
- **other**: TP=685 FP=156 TN=4667 FN=194
- **stroma**: TP=552 FP=187 TN=4756 FN=207

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 78 | 3 | 1 | 8 | 8 | 2 |
| cancer | 10 | 1133 | 0 | 84 | 38 | 64 |
| empty | 0 | 1 | 1614 | 0 | 5 | 1 |
| normal | 1 | 8 | 0 | 995 | 0 | 10 |
| other | 0 | 41 | 5 | 38 | 685 | 110 |
| stroma | 1 | 42 | 0 | 59 | 105 | 552 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8667    0.7800    0.8211       100
      cancer     0.9226    0.8525    0.8862      1329
       empty     0.9963    0.9957    0.9960      1621
      normal     0.8404    0.9813    0.9054      1014
       other     0.8145    0.7793    0.7965       879
      stroma     0.7470    0.7273    0.7370       759

    accuracy                         0.8869      5702
   macro avg     0.8646    0.8527    0.8570      5702
weighted avg     0.8879    0.8869    0.8860      5702

```