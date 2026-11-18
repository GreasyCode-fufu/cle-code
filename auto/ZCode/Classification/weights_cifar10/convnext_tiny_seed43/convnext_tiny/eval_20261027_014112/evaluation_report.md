# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_tiny_seed43/convnext_tiny/model_best.pth`
- 测试集: `./datasets/cifar10/val`  (共 5000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9800 |
| Accuracy | 0.9800 |
| Macro-AUROC (OvR) | 0.9997 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9800 |
| Balanced Acc | 0.9800 |
| Macro-Sensitivity | 0.9800 |
| Macro-Specificity | 0.9978 |
| Macro-Precision | 0.9801 |
| AUC-PR (Macro) | 0.9973 |
| Cohen's Kappa | 0.9778 |
| MCC | 0.9778 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.81M |
| Pure-Inference FPS | 616.2 (1.62 ms/张) |
| End-to-End FPS | 465.1 |
| 测试样本数 | 5000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9860 | 0.9993 | 0.9940 | 0.9900 | 0.9984 | 0.9999 | 0.9993 | 500 |
| automobile | 0.9920 | 0.9991 | 0.9920 | 0.9920 | 0.9991 | 0.9998 | 0.9986 | 500 |
| bird | 0.9880 | 0.9971 | 0.9744 | 0.9811 | 0.9987 | 0.9996 | 0.9976 | 500 |
| cat | 0.9620 | 0.9942 | 0.9487 | 0.9553 | 0.9958 | 0.9990 | 0.9914 | 500 |
| deer | 0.9740 | 0.9982 | 0.9838 | 0.9789 | 0.9971 | 0.9998 | 0.9977 | 500 |
| dog | 0.9540 | 0.9969 | 0.9715 | 0.9627 | 0.9949 | 0.9991 | 0.9934 | 500 |
| frog | 0.9780 | 0.9980 | 0.9819 | 0.9800 | 0.9976 | 0.9998 | 0.9987 | 500 |
| horse | 0.9860 | 0.9967 | 0.9705 | 0.9782 | 0.9984 | 0.9997 | 0.9980 | 500 |
| ship | 0.9940 | 0.9993 | 0.9940 | 0.9940 | 0.9993 | 1.0000 | 0.9996 | 500 |
| truck | 0.9860 | 0.9989 | 0.9900 | 0.9880 | 0.9984 | 0.9999 | 0.9988 | 500 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=493 FP=3 TN=4497 FN=7
- **automobile**: TP=496 FP=4 TN=4496 FN=4
- **bird**: TP=494 FP=13 TN=4487 FN=6
- **cat**: TP=481 FP=26 TN=4474 FN=19
- **deer**: TP=487 FP=8 TN=4492 FN=13
- **dog**: TP=477 FP=14 TN=4486 FN=23
- **frog**: TP=489 FP=9 TN=4491 FN=11
- **horse**: TP=493 FP=15 TN=4485 FN=7
- **ship**: TP=497 FP=3 TN=4497 FN=3
- **truck**: TP=493 FP=5 TN=4495 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 493 | 0 | 4 | 0 | 0 | 1 | 0 | 0 | 1 | 1 |
| automobile | 0 | 496 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| bird | 0 | 0 | 494 | 1 | 1 | 0 | 3 | 0 | 1 | 0 |
| cat | 0 | 0 | 5 | 481 | 1 | 7 | 4 | 2 | 0 | 0 |
| deer | 0 | 0 | 1 | 2 | 487 | 2 | 0 | 8 | 0 | 0 |
| dog | 0 | 0 | 0 | 17 | 1 | 477 | 1 | 4 | 0 | 0 |
| frog | 1 | 0 | 3 | 3 | 3 | 0 | 489 | 0 | 1 | 0 |
| horse | 0 | 0 | 0 | 2 | 2 | 3 | 0 | 493 | 0 | 0 |
| ship | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 497 | 0 |
| truck | 2 | 3 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 493 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9940    0.9860    0.9900       500
  automobile     0.9920    0.9920    0.9920       500
        bird     0.9744    0.9880    0.9811       500
         cat     0.9487    0.9620    0.9553       500
        deer     0.9838    0.9740    0.9789       500
         dog     0.9715    0.9540    0.9627       500
        frog     0.9819    0.9780    0.9800       500
       horse     0.9705    0.9860    0.9782       500
        ship     0.9940    0.9940    0.9940       500
       truck     0.9900    0.9860    0.9880       500

    accuracy                         0.9800      5000
   macro avg     0.9801    0.9800    0.9800      5000
weighted avg     0.9801    0.9800    0.9800      5000

```