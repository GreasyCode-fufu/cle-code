# mobilenet_v3_large 评估报告

- 权重: `weights_tcga_ep/mobilenet_v3_large_seed43/mobilenet_v3_large/model_best.pth`
- 测试集: `../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8207 |
| Accuracy | 0.8609 |
| Macro-AUROC (OvR) | 0.9740 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8635 |
| Balanced Acc | 0.8440 |
| Macro-Sensitivity | 0.8440 |
| Macro-Specificity | 0.9726 |
| Macro-Precision | 0.8077 |
| AUC-PR (Macro) | 0.8794 |
| Cohen's Kappa | 0.8252 |
| MCC | 0.8264 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.93M |
| Params | 4.21M |
| Pure-Inference FPS | 9135.8 (0.11 ms/张) |
| End-to-End FPS | 563.9 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8400 | 0.9909 | 0.6222 | 0.7149 | 0.9971 | 0.9918 | 0.8560 | 100 |
| cancer | 0.7848 | 0.9776 | 0.9141 | 0.8445 | 0.9373 | 0.9750 | 0.9295 | 1329 |
| empty | 0.9951 | 0.9966 | 0.9914 | 0.9932 | 0.9980 | 0.9996 | 0.9994 | 1621 |
| normal | 0.8945 | 0.9891 | 0.9468 | 0.9199 | 0.9774 | 0.9906 | 0.9738 | 1014 |
| other | 0.7167 | 0.9446 | 0.7023 | 0.7095 | 0.9482 | 0.9272 | 0.8114 | 879 |
| stroma | 0.8327 | 0.9369 | 0.6695 | 0.7422 | 0.9733 | 0.9597 | 0.7060 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=84 FP=51 TN=5551 FN=16
- **cancer**: TP=1043 FP=98 TN=4275 FN=286
- **empty**: TP=1613 FP=14 TN=4067 FN=8
- **normal**: TP=907 FP=51 TN=4637 FN=107
- **other**: TP=630 FP=267 TN=4556 FN=249
- **stroma**: TP=632 FP=312 TN=4631 FN=127

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 84 | 2 | 3 | 3 | 6 | 2 |
| cancer | 36 | 1043 | 0 | 29 | 137 | 84 |
| empty | 1 | 1 | 1613 | 0 | 4 | 2 |
| normal | 4 | 25 | 0 | 907 | 41 | 37 |
| other | 7 | 41 | 9 | 5 | 630 | 187 |
| stroma | 3 | 29 | 2 | 14 | 79 | 632 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6222    0.8400    0.7149       100
      cancer     0.9141    0.7848    0.8445      1329
       empty     0.9914    0.9951    0.9932      1621
      normal     0.9468    0.8945    0.9199      1014
       other     0.7023    0.7167    0.7095       879
      stroma     0.6695    0.8327    0.7422       759

    accuracy                         0.8609      5702
   macro avg     0.8077    0.8440    0.8207      5702
weighted avg     0.8716    0.8609    0.8635      5702

```