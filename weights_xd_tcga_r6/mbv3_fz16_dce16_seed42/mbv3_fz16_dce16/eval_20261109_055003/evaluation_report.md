# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce16_seed42/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8350 |
| Accuracy | 0.8679 |
| Macro-AUROC (OvR) | 0.9767 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8696 |
| Balanced Acc | 0.8429 |
| Macro-Sensitivity | 0.8429 |
| Macro-Specificity | 0.9736 |
| Macro-Precision | 0.8310 |
| AUC-PR (Macro) | 0.8865 |
| Cohen's Kappa | 0.8333 |
| MCC | 0.8339 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9105.7 (0.11 ms/张) |
| End-to-End FPS | 561.8 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8400 | 0.9946 | 0.7368 | 0.7850 | 0.9971 | 0.9930 | 0.8344 | 100 |
| cancer | 0.8886 | 0.9547 | 0.8564 | 0.8722 | 0.9658 | 0.9806 | 0.9506 | 1329 |
| empty | 0.9926 | 0.9988 | 0.9969 | 0.9947 | 0.9971 | 0.9999 | 0.9997 | 1621 |
| normal | 0.8314 | 0.9915 | 0.9547 | 0.8888 | 0.9645 | 0.9903 | 0.9682 | 1014 |
| other | 0.7486 | 0.9621 | 0.7824 | 0.7651 | 0.9545 | 0.9445 | 0.8555 | 879 |
| stroma | 0.7563 | 0.9399 | 0.6590 | 0.7043 | 0.9617 | 0.9516 | 0.7107 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=84 FP=30 TN=5572 FN=16
- **cancer**: TP=1181 FP=198 TN=4175 FN=148
- **empty**: TP=1609 FP=5 TN=4076 FN=12
- **normal**: TP=843 FP=40 TN=4648 FN=171
- **other**: TP=658 FP=183 TN=4640 FN=221
- **stroma**: TP=574 FP=297 TN=4646 FN=185

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 84 | 2 | 1 | 2 | 5 | 6 |
| cancer | 20 | 1181 | 1 | 19 | 39 | 69 |
| empty | 0 | 0 | 1609 | 0 | 10 | 2 |
| normal | 4 | 55 | 0 | 843 | 43 | 69 |
| other | 4 | 57 | 3 | 6 | 658 | 151 |
| stroma | 2 | 84 | 0 | 13 | 86 | 574 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7368    0.8400    0.7850       100
      cancer     0.8564    0.8886    0.8722      1329
       empty     0.9969    0.9926    0.9947      1621
      normal     0.9547    0.8314    0.8888      1014
       other     0.7824    0.7486    0.7651       879
      stroma     0.6590    0.7563    0.7043       759

    accuracy                         0.8679      5702
   macro avg     0.8310    0.8429    0.8350      5702
weighted avg     0.8740    0.8679    0.8696      5702

```