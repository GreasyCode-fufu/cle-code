# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_abl/convnext_v2c_seed44/convnext_v2c/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8634 |
| Accuracy | 0.8893 |
| Macro-AUROC (OvR) | 0.9800 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8918 |
| Balanced Acc | 0.8827 |
| Macro-Sensitivity | 0.8827 |
| Macro-Specificity | 0.9784 |
| Macro-Precision | 0.8542 |
| AUC-PR (Macro) | 0.9149 |
| Cohen's Kappa | 0.8609 |
| MCC | 0.8627 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.26M |
| Pure-Inference FPS | 1786.8 (0.56 ms/张) |
| End-to-End FPS | 488.1 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.9100 | 0.9945 | 0.7459 | 0.8198 | 0.9984 | 0.9973 | 0.9136 | 100 |
| cancer | 0.8119 | 0.9865 | 0.9482 | 0.8747 | 0.9452 | 0.9757 | 0.9483 | 1329 |
| empty | 0.9920 | 0.9988 | 0.9969 | 0.9944 | 0.9968 | 0.9998 | 0.9995 | 1621 |
| normal | 0.9497 | 0.9915 | 0.9601 | 0.9549 | 0.9891 | 0.9969 | 0.9886 | 1014 |
| other | 0.7554 | 0.9691 | 0.8167 | 0.7849 | 0.9560 | 0.9461 | 0.8604 | 879 |
| stroma | 0.8775 | 0.9298 | 0.6575 | 0.7517 | 0.9802 | 0.9645 | 0.7787 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=91 FP=31 TN=5571 FN=9
- **cancer**: TP=1079 FP=59 TN=4314 FN=250
- **empty**: TP=1608 FP=5 TN=4076 FN=13
- **normal**: TP=963 FP=40 TN=4648 FN=51
- **other**: TP=664 FP=149 TN=4674 FN=215
- **stroma**: TP=666 FP=347 TN=4596 FN=93

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 91 | 1 | 0 | 2 | 5 | 1 |
| cancer | 21 | 1079 | 0 | 11 | 74 | 144 |
| empty | 1 | 1 | 1608 | 0 | 9 | 2 |
| normal | 4 | 15 | 0 | 963 | 3 | 29 |
| other | 1 | 29 | 5 | 9 | 664 | 171 |
| stroma | 4 | 13 | 0 | 18 | 58 | 666 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7459    0.9100    0.8198       100
      cancer     0.9482    0.8119    0.8747      1329
       empty     0.9969    0.9920    0.9944      1621
      normal     0.9601    0.9497    0.9549      1014
       other     0.8167    0.7554    0.7849       879
      stroma     0.6575    0.8775    0.7517       759

    accuracy                         0.8893      5702
   macro avg     0.8542    0.8827    0.8634      5702
weighted avg     0.9016    0.8893    0.8918      5702

```