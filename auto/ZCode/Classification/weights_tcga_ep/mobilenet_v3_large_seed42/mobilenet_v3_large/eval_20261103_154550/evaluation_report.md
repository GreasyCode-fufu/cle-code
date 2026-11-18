# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/mobilenet_v3_large_seed42/mobilenet_v3_large/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8345 |
| Accuracy | 0.8720 |
| Macro-AUROC (OvR) | 0.9785 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8737 |
| Balanced Acc | 0.8459 |
| Macro-Sensitivity | 0.8459 |
| Macro-Specificity | 0.9747 |
| Macro-Precision | 0.8278 |
| AUC-PR (Macro) | 0.8919 |
| Cohen's Kappa | 0.8388 |
| MCC | 0.8397 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.93M |
| Params | 4.21M |
| Pure-Inference FPS | 3876.3 (0.26 ms/张) |
| End-to-End FPS | 567.7 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8200 | 0.9939 | 0.7069 | 0.7593 | 0.9968 | 0.9946 | 0.8722 | 100 |
| cancer | 0.8232 | 0.9792 | 0.9232 | 0.8703 | 0.9480 | 0.9848 | 0.9577 | 1329 |
| empty | 0.9883 | 0.9993 | 0.9981 | 0.9932 | 0.9954 | 0.9996 | 0.9993 | 1621 |
| normal | 0.9389 | 0.9874 | 0.9416 | 0.9402 | 0.9868 | 0.9945 | 0.9808 | 1014 |
| other | 0.8294 | 0.9303 | 0.6845 | 0.7500 | 0.9677 | 0.9523 | 0.8421 | 879 |
| stroma | 0.6759 | 0.9581 | 0.7125 | 0.6937 | 0.9506 | 0.9449 | 0.6994 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=82 FP=34 TN=5568 FN=18
- **cancer**: TP=1094 FP=91 TN=4282 FN=235
- **empty**: TP=1602 FP=3 TN=4078 FN=19
- **normal**: TP=952 FP=59 TN=4629 FN=62
- **other**: TP=729 FP=336 TN=4487 FN=150
- **stroma**: TP=513 FP=207 TN=4736 FN=246

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 82 | 1 | 1 | 6 | 7 | 3 |
| cancer | 22 | 1094 | 0 | 35 | 105 | 73 |
| empty | 0 | 1 | 1602 | 0 | 16 | 2 |
| normal | 5 | 24 | 0 | 952 | 8 | 25 |
| other | 2 | 35 | 2 | 7 | 729 | 104 |
| stroma | 5 | 30 | 0 | 11 | 200 | 513 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7069    0.8200    0.7593       100
      cancer     0.9232    0.8232    0.8703      1329
       empty     0.9981    0.9883    0.9932      1621
      normal     0.9416    0.9389    0.9402      1014
       other     0.6845    0.8294    0.7500       879
      stroma     0.7125    0.6759    0.6937       759

    accuracy                         0.8720      5702
   macro avg     0.8278    0.8459    0.8345      5702
weighted avg     0.8791    0.8720    0.8737      5702

```