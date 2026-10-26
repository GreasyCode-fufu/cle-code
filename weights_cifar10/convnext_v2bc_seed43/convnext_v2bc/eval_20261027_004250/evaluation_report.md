# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_v2bc_seed43/convnext_v2bc/model_best.pth`
- 测试集: `./datasets/cifar10/test`  (共 10000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9762 |
| Accuracy | 0.9762 |
| Macro-AUROC (OvR) | 0.9992 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9762 |
| Balanced Acc | 0.9762 |
| Macro-Sensitivity | 0.9762 |
| Macro-Specificity | 0.9974 |
| Macro-Precision | 0.9762 |
| AUC-PR (Macro) | 0.9952 |
| Cohen's Kappa | 0.9736 |
| MCC | 0.9736 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.68M |
| Pure-Inference FPS | 595.5 (1.68 ms/张) |
| End-to-End FPS | 453.0 |
| 测试样本数 | 10000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9820 | 0.9979 | 0.9810 | 0.9815 | 0.9980 | 0.9998 | 0.9982 | 1000 |
| automobile | 0.9910 | 0.9982 | 0.9841 | 0.9875 | 0.9990 | 0.9992 | 0.9974 | 1000 |
| bird | 0.9750 | 0.9974 | 0.9770 | 0.9760 | 0.9972 | 0.9990 | 0.9959 | 1000 |
| cat | 0.9320 | 0.9941 | 0.9462 | 0.9390 | 0.9925 | 0.9973 | 0.9832 | 1000 |
| deer | 0.9790 | 0.9977 | 0.9790 | 0.9790 | 0.9977 | 0.9997 | 0.9974 | 1000 |
| dog | 0.9550 | 0.9948 | 0.9531 | 0.9540 | 0.9950 | 0.9981 | 0.9886 | 1000 |
| frog | 0.9940 | 0.9978 | 0.9803 | 0.9871 | 0.9993 | 0.9997 | 0.9966 | 1000 |
| horse | 0.9800 | 0.9989 | 0.9899 | 0.9849 | 0.9978 | 0.9999 | 0.9990 | 1000 |
| ship | 0.9890 | 0.9983 | 0.9851 | 0.9870 | 0.9988 | 0.9998 | 0.9988 | 1000 |
| truck | 0.9850 | 0.9984 | 0.9860 | 0.9855 | 0.9983 | 0.9996 | 0.9967 | 1000 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=982 FP=19 TN=8981 FN=18
- **automobile**: TP=991 FP=16 TN=8984 FN=9
- **bird**: TP=975 FP=23 TN=8977 FN=25
- **cat**: TP=932 FP=53 TN=8947 FN=68
- **deer**: TP=979 FP=21 TN=8979 FN=21
- **dog**: TP=955 FP=47 TN=8953 FN=45
- **frog**: TP=994 FP=20 TN=8980 FN=6
- **horse**: TP=980 FP=10 TN=8990 FN=20
- **ship**: TP=989 FP=15 TN=8985 FN=11
- **truck**: TP=985 FP=14 TN=8986 FN=15

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 982 | 1 | 2 | 0 | 0 | 0 | 0 | 1 | 12 | 2 |
| automobile | 1 | 991 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| bird | 3 | 0 | 975 | 6 | 6 | 2 | 7 | 0 | 1 | 0 |
| cat | 2 | 0 | 7 | 932 | 5 | 39 | 8 | 5 | 1 | 1 |
| deer | 1 | 0 | 6 | 6 | 979 | 1 | 3 | 4 | 0 | 0 |
| dog | 0 | 0 | 1 | 38 | 5 | 955 | 1 | 0 | 0 | 0 |
| frog | 2 | 0 | 1 | 0 | 1 | 1 | 994 | 0 | 0 | 1 |
| horse | 2 | 0 | 5 | 3 | 4 | 4 | 1 | 980 | 1 | 0 |
| ship | 8 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 989 | 2 |
| truck | 0 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 985 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9810    0.9820    0.9815      1000
  automobile     0.9841    0.9910    0.9875      1000
        bird     0.9770    0.9750    0.9760      1000
         cat     0.9462    0.9320    0.9390      1000
        deer     0.9790    0.9790    0.9790      1000
         dog     0.9531    0.9550    0.9540      1000
        frog     0.9803    0.9940    0.9871      1000
       horse     0.9899    0.9800    0.9849      1000
        ship     0.9851    0.9890    0.9870      1000
       truck     0.9860    0.9850    0.9855      1000

    accuracy                         0.9762     10000
   macro avg     0.9762    0.9762    0.9762     10000
weighted avg     0.9762    0.9762    0.9762     10000

```