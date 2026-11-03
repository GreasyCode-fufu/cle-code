# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/ghostnet_seed44/ghostnet/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8603 |
| Accuracy | 0.8843 |
| Macro-AUROC (OvR) | 0.9809 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8869 |
| Balanced Acc | 0.8760 |
| Macro-Sensitivity | 0.8760 |
| Macro-Specificity | 0.9773 |
| Macro-Precision | 0.8514 |
| AUC-PR (Macro) | 0.9118 |
| Cohen's Kappa | 0.8544 |
| MCC | 0.8556 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.59M |
| FLOPs | 309.17M |
| Params | 3.91M |
| Pure-Inference FPS | 4397.5 (0.23 ms/张) |
| End-to-End FPS | 570.6 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.9200 | 0.9954 | 0.7797 | 0.8440 | 0.9986 | 0.9977 | 0.9306 | 100 |
| cancer | 0.8397 | 0.9867 | 0.9506 | 0.8917 | 0.9530 | 0.9864 | 0.9632 | 1329 |
| empty | 0.9914 | 0.9983 | 0.9957 | 0.9935 | 0.9966 | 0.9999 | 0.9997 | 1621 |
| normal | 0.9310 | 0.9904 | 0.9545 | 0.9426 | 0.9851 | 0.9937 | 0.9817 | 1014 |
| other | 0.7372 | 0.9602 | 0.7714 | 0.7539 | 0.9525 | 0.9489 | 0.8319 | 879 |
| stroma | 0.8366 | 0.9328 | 0.6567 | 0.7358 | 0.9738 | 0.9588 | 0.7638 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=92 FP=26 TN=5576 FN=8
- **cancer**: TP=1116 FP=58 TN=4315 FN=213
- **empty**: TP=1607 FP=7 TN=4074 FN=14
- **normal**: TP=944 FP=45 TN=4643 FN=70
- **other**: TP=648 FP=192 TN=4631 FN=231
- **stroma**: TP=635 FP=332 TN=4611 FN=124

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 92 | 1 | 0 | 3 | 2 | 2 |
| cancer | 17 | 1116 | 0 | 20 | 56 | 120 |
| empty | 0 | 1 | 1607 | 0 | 11 | 2 |
| normal | 4 | 18 | 0 | 944 | 25 | 23 |
| other | 2 | 29 | 6 | 9 | 648 | 185 |
| stroma | 3 | 9 | 1 | 13 | 98 | 635 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7797    0.9200    0.8440       100
      cancer     0.9506    0.8397    0.8917      1329
       empty     0.9957    0.9914    0.9935      1621
      normal     0.9545    0.9310    0.9426      1014
       other     0.7714    0.7372    0.7539       879
      stroma     0.6567    0.8366    0.7358       759

    accuracy                         0.8843      5702
   macro avg     0.8514    0.8760    0.8603      5702
weighted avg     0.8944    0.8843    0.8869      5702

```