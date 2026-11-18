# deit_small_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/deit_small_patch16_224_seed42/deit_small_patch16_224/model_best.pth`
- 测试集: `./datasets/cifar10/val`  (共 5000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9712 |
| Accuracy | 0.9712 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9712 |
| Balanced Acc | 0.9712 |
| Macro-Sensitivity | 0.9712 |
| Macro-Specificity | 0.9968 |
| Macro-Precision | 0.9713 |
| AUC-PR (Macro) | 0.9949 |
| Cohen's Kappa | 0.9680 |
| MCC | 0.9680 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.25G |
| FLOPs | 8.50G |
| Params | 21.59M |
| Pure-Inference FPS | 969.5 (1.03 ms/张) |
| End-to-End FPS | 639.7 |
| 测试样本数 | 5000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9760 | 0.9976 | 0.9780 | 0.9770 | 0.9973 | 0.9995 | 0.9966 | 500 |
| automobile | 0.9840 | 0.9973 | 0.9762 | 0.9801 | 0.9982 | 0.9998 | 0.9984 | 500 |
| bird | 0.9620 | 0.9976 | 0.9776 | 0.9698 | 0.9958 | 0.9993 | 0.9951 | 500 |
| cat | 0.9580 | 0.9922 | 0.9319 | 0.9448 | 0.9953 | 0.9982 | 0.9862 | 500 |
| deer | 0.9780 | 0.9969 | 0.9722 | 0.9751 | 0.9976 | 0.9995 | 0.9962 | 500 |
| dog | 0.9260 | 0.9960 | 0.9626 | 0.9439 | 0.9918 | 0.9974 | 0.9851 | 500 |
| frog | 0.9860 | 0.9987 | 0.9880 | 0.9870 | 0.9984 | 0.9998 | 0.9983 | 500 |
| horse | 0.9840 | 0.9964 | 0.9685 | 0.9762 | 0.9982 | 0.9998 | 0.9982 | 500 |
| ship | 0.9780 | 0.9987 | 0.9879 | 0.9829 | 0.9976 | 0.9998 | 0.9983 | 500 |
| truck | 0.9800 | 0.9967 | 0.9703 | 0.9751 | 0.9978 | 0.9995 | 0.9969 | 500 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=488 FP=11 TN=4489 FN=12
- **automobile**: TP=492 FP=12 TN=4488 FN=8
- **bird**: TP=481 FP=11 TN=4489 FN=19
- **cat**: TP=479 FP=35 TN=4465 FN=21
- **deer**: TP=489 FP=14 TN=4486 FN=11
- **dog**: TP=463 FP=18 TN=4482 FN=37
- **frog**: TP=493 FP=6 TN=4494 FN=7
- **horse**: TP=492 FP=16 TN=4484 FN=8
- **ship**: TP=489 FP=6 TN=4494 FN=11
- **truck**: TP=490 FP=15 TN=4485 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 488 | 1 | 2 | 2 | 1 | 0 | 0 | 0 | 5 | 1 |
| automobile | 0 | 492 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| bird | 3 | 0 | 481 | 5 | 8 | 1 | 0 | 1 | 1 | 0 |
| cat | 2 | 0 | 2 | 479 | 0 | 13 | 2 | 0 | 0 | 2 |
| deer | 0 | 0 | 1 | 2 | 489 | 1 | 0 | 7 | 0 | 0 |
| dog | 1 | 0 | 1 | 22 | 2 | 463 | 3 | 8 | 0 | 0 |
| frog | 1 | 0 | 2 | 3 | 1 | 0 | 493 | 0 | 0 | 0 |
| horse | 0 | 0 | 2 | 0 | 2 | 3 | 0 | 492 | 0 | 1 |
| ship | 4 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 489 | 3 |
| truck | 0 | 9 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 490 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9780    0.9760    0.9770       500
  automobile     0.9762    0.9840    0.9801       500
        bird     0.9776    0.9620    0.9698       500
         cat     0.9319    0.9580    0.9448       500
        deer     0.9722    0.9780    0.9751       500
         dog     0.9626    0.9260    0.9439       500
        frog     0.9880    0.9860    0.9870       500
       horse     0.9685    0.9840    0.9762       500
        ship     0.9879    0.9780    0.9829       500
       truck     0.9703    0.9800    0.9751       500

    accuracy                         0.9712      5000
   macro avg     0.9713    0.9712    0.9712      5000
weighted avg     0.9713    0.9712    0.9712      5000

```