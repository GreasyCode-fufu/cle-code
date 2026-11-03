# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/ghostnet_seed43/ghostnet/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8162 |
| Accuracy | 0.8695 |
| Macro-AUROC (OvR) | 0.9716 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8719 |
| Balanced Acc | 0.8417 |
| Macro-Sensitivity | 0.8417 |
| Macro-Specificity | 0.9743 |
| Macro-Precision | 0.8013 |
| AUC-PR (Macro) | 0.8724 |
| Cohen's Kappa | 0.8356 |
| MCC | 0.8358 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.59M |
| FLOPs | 309.17M |
| Params | 3.91M |
| Pure-Inference FPS | 4313.2 (0.23 ms/张) |
| End-to-End FPS | 566.7 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8200 | 0.9873 | 0.5359 | 0.6482 | 0.9968 | 0.9802 | 0.7881 | 100 |
| cancer | 0.8555 | 0.9668 | 0.8869 | 0.8709 | 0.9566 | 0.9755 | 0.9375 | 1329 |
| empty | 0.9926 | 0.9985 | 0.9963 | 0.9944 | 0.9971 | 0.9995 | 0.9988 | 1621 |
| normal | 0.9063 | 0.9945 | 0.9725 | 0.9382 | 0.9800 | 0.9925 | 0.9774 | 1014 |
| other | 0.7565 | 0.9478 | 0.7252 | 0.7405 | 0.9553 | 0.9306 | 0.8290 | 879 |
| stroma | 0.7194 | 0.9506 | 0.6911 | 0.7050 | 0.9566 | 0.9511 | 0.7036 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=82 FP=71 TN=5531 FN=18
- **cancer**: TP=1137 FP=145 TN=4228 FN=192
- **empty**: TP=1609 FP=6 TN=4075 FN=12
- **normal**: TP=919 FP=26 TN=4662 FN=95
- **other**: TP=665 FP=252 TN=4571 FN=214
- **stroma**: TP=546 FP=244 TN=4699 FN=213

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 82 | 2 | 0 | 1 | 10 | 5 |
| cancer | 6 | 1137 | 0 | 7 | 95 | 84 |
| empty | 0 | 1 | 1609 | 0 | 9 | 2 |
| normal | 3 | 17 | 0 | 919 | 47 | 28 |
| other | 19 | 60 | 5 | 5 | 665 | 125 |
| stroma | 43 | 65 | 1 | 13 | 91 | 546 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.5359    0.8200    0.6482       100
      cancer     0.8869    0.8555    0.8709      1329
       empty     0.9963    0.9926    0.9944      1621
      normal     0.9725    0.9063    0.9382      1014
       other     0.7252    0.7565    0.7405       879
      stroma     0.6911    0.7194    0.7050       759

    accuracy                         0.8695      5702
   macro avg     0.8013    0.8417    0.8162      5702
weighted avg     0.8761    0.8695    0.8719      5702

```