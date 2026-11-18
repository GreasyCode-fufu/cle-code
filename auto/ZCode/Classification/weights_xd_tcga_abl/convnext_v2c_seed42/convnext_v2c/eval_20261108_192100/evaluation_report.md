# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_abl/convnext_v2c_seed42/convnext_v2c/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8784 |
| Accuracy | 0.9065 |
| Macro-AUROC (OvR) | 0.9806 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9078 |
| Balanced Acc | 0.8748 |
| Macro-Sensitivity | 0.8748 |
| Macro-Specificity | 0.9815 |
| Macro-Precision | 0.8894 |
| AUC-PR (Macro) | 0.9162 |
| Cohen's Kappa | 0.8821 |
| MCC | 0.8833 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.26M |
| Pure-Inference FPS | 888.4 (1.13 ms/张) |
| End-to-End FPS | 378.1 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7800 | 0.9980 | 0.8764 | 0.8254 | 0.9961 | 0.9973 | 0.9011 | 100 |
| cancer | 0.8947 | 0.9828 | 0.9407 | 0.9171 | 0.9685 | 0.9757 | 0.9530 | 1329 |
| empty | 0.9963 | 0.9978 | 0.9945 | 0.9954 | 0.9985 | 0.9996 | 0.9991 | 1621 |
| normal | 0.9467 | 0.9900 | 0.9533 | 0.9500 | 0.9885 | 0.9910 | 0.9804 | 1014 |
| other | 0.7418 | 0.9818 | 0.8811 | 0.8054 | 0.9543 | 0.9515 | 0.8872 | 879 |
| stroma | 0.8893 | 0.9387 | 0.6902 | 0.7772 | 0.9822 | 0.9681 | 0.7765 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=78 FP=11 TN=5591 FN=22
- **cancer**: TP=1189 FP=75 TN=4298 FN=140
- **empty**: TP=1615 FP=9 TN=4072 FN=6
- **normal**: TP=960 FP=47 TN=4641 FN=54
- **other**: TP=652 FP=88 TN=4735 FN=227
- **stroma**: TP=675 FP=303 TN=4640 FN=84

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 78 | 2 | 1 | 3 | 10 | 6 |
| cancer | 8 | 1189 | 1 | 6 | 23 | 102 |
| empty | 1 | 1 | 1615 | 0 | 3 | 1 |
| normal | 2 | 10 | 0 | 960 | 10 | 32 |
| other | 0 | 44 | 7 | 14 | 652 | 162 |
| stroma | 0 | 18 | 0 | 24 | 42 | 675 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8764    0.7800    0.8254       100
      cancer     0.9407    0.8947    0.9171      1329
       empty     0.9945    0.9963    0.9954      1621
      normal     0.9533    0.9467    0.9500      1014
       other     0.8811    0.7418    0.8054       879
      stroma     0.6902    0.8893    0.7772       759

    accuracy                         0.9065      5702
   macro avg     0.8894    0.8748    0.8784      5702
weighted avg     0.9146    0.9065    0.9078      5702

```