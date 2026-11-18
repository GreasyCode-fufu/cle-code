# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_tiny_seed42/convnext_tiny/model_best.pth`
- 测试集: `./datasets/cifar10/test`  (共 10000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9750 |
| Accuracy | 0.9750 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9750 |
| Balanced Acc | 0.9750 |
| Macro-Sensitivity | 0.9750 |
| Macro-Specificity | 0.9972 |
| Macro-Precision | 0.9750 |
| AUC-PR (Macro) | 0.9952 |
| Cohen's Kappa | 0.9722 |
| MCC | 0.9722 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.81M |
| Pure-Inference FPS | 604.5 (1.65 ms/张) |
| End-to-End FPS | 455.1 |
| 测试样本数 | 10000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9840 | 0.9978 | 0.9801 | 0.9820 | 0.9982 | 0.9998 | 0.9981 | 1000 |
| automobile | 0.9860 | 0.9979 | 0.9811 | 0.9835 | 0.9984 | 0.9990 | 0.9965 | 1000 |
| bird | 0.9750 | 0.9967 | 0.9701 | 0.9726 | 0.9972 | 0.9995 | 0.9966 | 1000 |
| cat | 0.9420 | 0.9938 | 0.9439 | 0.9429 | 0.9936 | 0.9976 | 0.9841 | 1000 |
| deer | 0.9770 | 0.9971 | 0.9741 | 0.9755 | 0.9974 | 0.9997 | 0.9973 | 1000 |
| dog | 0.9510 | 0.9957 | 0.9606 | 0.9558 | 0.9946 | 0.9987 | 0.9905 | 1000 |
| frog | 0.9880 | 0.9981 | 0.9831 | 0.9855 | 0.9987 | 0.9999 | 0.9983 | 1000 |
| horse | 0.9820 | 0.9987 | 0.9879 | 0.9850 | 0.9980 | 0.9998 | 0.9987 | 1000 |
| ship | 0.9880 | 0.9983 | 0.9850 | 0.9865 | 0.9987 | 0.9999 | 0.9990 | 1000 |
| truck | 0.9770 | 0.9982 | 0.9839 | 0.9804 | 0.9974 | 0.9993 | 0.9929 | 1000 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=984 FP=20 TN=8980 FN=16
- **automobile**: TP=986 FP=19 TN=8981 FN=14
- **bird**: TP=975 FP=30 TN=8970 FN=25
- **cat**: TP=942 FP=56 TN=8944 FN=58
- **deer**: TP=977 FP=26 TN=8974 FN=23
- **dog**: TP=951 FP=39 TN=8961 FN=49
- **frog**: TP=988 FP=17 TN=8983 FN=12
- **horse**: TP=982 FP=12 TN=8988 FN=18
- **ship**: TP=988 FP=15 TN=8985 FN=12
- **truck**: TP=977 FP=16 TN=8984 FN=23

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 984 | 1 | 2 | 1 | 0 | 0 | 1 | 1 | 9 | 1 |
| automobile | 1 | 986 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 12 |
| bird | 2 | 0 | 975 | 7 | 7 | 3 | 4 | 2 | 0 | 0 |
| cat | 2 | 0 | 7 | 942 | 7 | 30 | 9 | 1 | 1 | 1 |
| deer | 0 | 0 | 10 | 4 | 977 | 1 | 3 | 5 | 0 | 0 |
| dog | 0 | 0 | 3 | 37 | 6 | 951 | 0 | 3 | 0 | 0 |
| frog | 1 | 0 | 4 | 4 | 1 | 1 | 988 | 0 | 0 | 1 |
| horse | 2 | 0 | 3 | 3 | 5 | 4 | 0 | 982 | 1 | 0 |
| ship | 9 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 988 | 1 |
| truck | 3 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 977 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9801    0.9840    0.9820      1000
  automobile     0.9811    0.9860    0.9835      1000
        bird     0.9701    0.9750    0.9726      1000
         cat     0.9439    0.9420    0.9429      1000
        deer     0.9741    0.9770    0.9755      1000
         dog     0.9606    0.9510    0.9558      1000
        frog     0.9831    0.9880    0.9855      1000
       horse     0.9879    0.9820    0.9850      1000
        ship     0.9850    0.9880    0.9865      1000
       truck     0.9839    0.9770    0.9804      1000

    accuracy                         0.9750     10000
   macro avg     0.9750    0.9750    0.9750     10000
weighted avg     0.9750    0.9750    0.9750     10000

```