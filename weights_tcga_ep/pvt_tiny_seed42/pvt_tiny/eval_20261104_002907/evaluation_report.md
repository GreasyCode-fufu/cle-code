# pvt_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/pvt_tiny_seed42/pvt_tiny/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8139 |
| Accuracy | 0.8478 |
| Macro-AUROC (OvR) | 0.9733 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8493 |
| Balanced Acc | 0.8141 |
| Macro-Sensitivity | 0.8141 |
| Macro-Specificity | 0.9701 |
| Macro-Precision | 0.8352 |
| AUC-PR (Macro) | 0.8819 |
| Cohen's Kappa | 0.8090 |
| MCC | 0.8137 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.86G |
| FLOPs | 3.73G |
| Params | 12.33M |
| Pure-Inference FPS | 1376.2 (0.73 ms/张) |
| End-to-End FPS | 439.0 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.6700 | 0.9977 | 0.8375 | 0.7444 | 0.9941 | 0.9905 | 0.8211 | 100 |
| cancer | 0.6840 | 0.9838 | 0.9276 | 0.7874 | 0.9111 | 0.9497 | 0.9031 | 1329 |
| empty | 0.9926 | 0.9978 | 0.9944 | 0.9935 | 0.9971 | 0.9999 | 0.9998 | 1621 |
| normal | 0.9418 | 0.9765 | 0.8967 | 0.9187 | 0.9873 | 0.9935 | 0.9765 | 1014 |
| other | 0.6883 | 0.9614 | 0.7649 | 0.7246 | 0.9442 | 0.9435 | 0.8131 | 879 |
| stroma | 0.9078 | 0.9031 | 0.5899 | 0.7151 | 0.9846 | 0.9629 | 0.7780 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=67 FP=13 TN=5589 FN=33
- **cancer**: TP=909 FP=71 TN=4302 FN=420
- **empty**: TP=1609 FP=9 TN=4072 FN=12
- **normal**: TP=955 FP=110 TN=4578 FN=59
- **other**: TP=605 FP=186 TN=4637 FN=274
- **stroma**: TP=689 FP=479 TN=4464 FN=70

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 67 | 8 | 1 | 2 | 13 | 9 |
| cancer | 7 | 909 | 0 | 49 | 144 | 220 |
| empty | 0 | 0 | 1609 | 0 | 10 | 2 |
| normal | 4 | 10 | 0 | 955 | 4 | 41 |
| other | 1 | 42 | 8 | 16 | 605 | 207 |
| stroma | 1 | 11 | 0 | 43 | 15 | 689 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8375    0.6700    0.7444       100
      cancer     0.9276    0.6840    0.7874      1329
       empty     0.9944    0.9926    0.9935      1621
      normal     0.8967    0.9418    0.9187      1014
       other     0.7649    0.6883    0.7246       879
      stroma     0.5899    0.9078    0.7151       759

    accuracy                         0.8478      5702
   macro avg     0.8352    0.8141    0.8139      5702
weighted avg     0.8695    0.8478    0.8493      5702

```