# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_v2c_seed42/convnext_v2c/model_best.pth`
- 测试集: `./datasets/cifar10/val`  (共 5000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9820 |
| Accuracy | 0.9820 |
| Macro-AUROC (OvR) | 0.9997 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9820 |
| Balanced Acc | 0.9820 |
| Macro-Sensitivity | 0.9820 |
| Macro-Specificity | 0.9980 |
| Macro-Precision | 0.9821 |
| AUC-PR (Macro) | 0.9978 |
| Cohen's Kappa | 0.9800 |
| MCC | 0.9800 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.27M |
| Pure-Inference FPS | 644.3 (1.55 ms/张) |
| End-to-End FPS | 481.3 |
| 测试样本数 | 5000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9900 | 0.9976 | 0.9783 | 0.9841 | 0.9989 | 0.9999 | 0.9988 | 500 |
| automobile | 0.9880 | 0.9989 | 0.9900 | 0.9890 | 0.9987 | 0.9999 | 0.9991 | 500 |
| bird | 0.9780 | 0.9978 | 0.9800 | 0.9790 | 0.9976 | 0.9996 | 0.9970 | 500 |
| cat | 0.9720 | 0.9951 | 0.9567 | 0.9643 | 0.9969 | 0.9992 | 0.9938 | 500 |
| deer | 0.9820 | 0.9989 | 0.9899 | 0.9859 | 0.9980 | 0.9997 | 0.9980 | 500 |
| dog | 0.9660 | 0.9978 | 0.9797 | 0.9728 | 0.9962 | 0.9995 | 0.9958 | 500 |
| frog | 0.9740 | 0.9989 | 0.9898 | 0.9819 | 0.9971 | 0.9999 | 0.9990 | 500 |
| horse | 0.9900 | 0.9973 | 0.9763 | 0.9831 | 0.9989 | 0.9997 | 0.9978 | 500 |
| ship | 0.9920 | 0.9993 | 0.9940 | 0.9930 | 0.9991 | 1.0000 | 0.9997 | 500 |
| truck | 0.9880 | 0.9984 | 0.9860 | 0.9870 | 0.9987 | 0.9999 | 0.9992 | 500 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=495 FP=11 TN=4489 FN=5
- **automobile**: TP=494 FP=5 TN=4495 FN=6
- **bird**: TP=489 FP=10 TN=4490 FN=11
- **cat**: TP=486 FP=22 TN=4478 FN=14
- **deer**: TP=491 FP=5 TN=4495 FN=9
- **dog**: TP=483 FP=10 TN=4490 FN=17
- **frog**: TP=487 FP=5 TN=4495 FN=13
- **horse**: TP=495 FP=12 TN=4488 FN=5
- **ship**: TP=496 FP=3 TN=4497 FN=4
- **truck**: TP=494 FP=7 TN=4493 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 495 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| automobile | 1 | 494 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| bird | 4 | 0 | 489 | 3 | 1 | 0 | 2 | 0 | 1 | 0 |
| cat | 1 | 0 | 1 | 486 | 1 | 8 | 1 | 1 | 0 | 1 |
| deer | 0 | 0 | 1 | 0 | 491 | 1 | 0 | 7 | 0 | 0 |
| dog | 0 | 0 | 0 | 12 | 0 | 483 | 1 | 4 | 0 | 0 |
| frog | 1 | 0 | 3 | 5 | 3 | 0 | 487 | 0 | 1 | 0 |
| horse | 0 | 0 | 2 | 2 | 0 | 1 | 0 | 495 | 0 | 0 |
| ship | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 496 | 0 |
| truck | 1 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 494 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9783    0.9900    0.9841       500
  automobile     0.9900    0.9880    0.9890       500
        bird     0.9800    0.9780    0.9790       500
         cat     0.9567    0.9720    0.9643       500
        deer     0.9899    0.9820    0.9859       500
         dog     0.9797    0.9660    0.9728       500
        frog     0.9898    0.9740    0.9819       500
       horse     0.9763    0.9900    0.9831       500
        ship     0.9940    0.9920    0.9930       500
       truck     0.9860    0.9880    0.9870       500

    accuracy                         0.9820      5000
   macro avg     0.9821    0.9820    0.9820      5000
weighted avg     0.9821    0.9820    0.9820      5000

```