# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_egb_scg_seed42/convnext_egb_scg/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8465 |
| Accuracy | 0.8927 |
| Macro-AUROC (OvR) | 0.9819 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8936 |
| Balanced Acc | 0.8735 |
| Macro-Sensitivity | 0.8735 |
| Macro-Specificity | 0.9790 |
| Macro-Precision | 0.8377 |
| AUC-PR (Macro) | 0.9019 |
| Cohen's Kappa | 0.8650 |
| MCC | 0.8672 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.29M |
| Pure-Inference FPS | 343.3 (2.91 ms/张) |
| End-to-End FPS | 227.2 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8400 | 0.9896 | 0.5915 | 0.6942 | 0.9971 | 0.9926 | 0.8203 | 100 |
| cancer | 0.8427 | 0.9838 | 0.9404 | 0.8889 | 0.9537 | 0.9772 | 0.9451 | 1329 |
| empty | 0.9963 | 0.9963 | 0.9908 | 0.9935 | 0.9985 | 0.9999 | 0.9998 | 1621 |
| normal | 0.9596 | 0.9878 | 0.9447 | 0.9521 | 0.9912 | 0.9974 | 0.9883 | 1014 |
| other | 0.6815 | 0.9828 | 0.8783 | 0.7675 | 0.9442 | 0.9528 | 0.8647 | 879 |
| stroma | 0.9209 | 0.9336 | 0.6806 | 0.7828 | 0.9872 | 0.9715 | 0.7932 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=84 FP=58 TN=5544 FN=16
- **cancer**: TP=1120 FP=71 TN=4302 FN=209
- **empty**: TP=1615 FP=15 TN=4066 FN=6
- **normal**: TP=973 FP=57 TN=4631 FN=41
- **other**: TP=599 FP=83 TN=4740 FN=280
- **stroma**: TP=699 FP=328 TN=4615 FN=60

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 84 | 0 | 1 | 6 | 6 | 3 |
| cancer | 46 | 1120 | 1 | 7 | 47 | 108 |
| empty | 0 | 0 | 1615 | 0 | 4 | 2 |
| normal | 2 | 14 | 0 | 973 | 5 | 20 |
| other | 8 | 45 | 13 | 19 | 599 | 195 |
| stroma | 2 | 12 | 0 | 25 | 21 | 699 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.5915    0.8400    0.6942       100
      cancer     0.9404    0.8427    0.8889      1329
       empty     0.9908    0.9963    0.9935      1621
      normal     0.9447    0.9596    0.9521      1014
       other     0.8783    0.6815    0.7675       879
      stroma     0.6806    0.9209    0.7828       759

    accuracy                         0.8927      5702
   macro avg     0.8377    0.8735    0.8465      5702
weighted avg     0.9052    0.8927    0.8936      5702

```