# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_v2c_seed42/convnext_v2c/model_best.pth`
- 测试集: `./datasets/cifar10/test`  (共 10000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9771 |
| Accuracy | 0.9771 |
| Macro-AUROC (OvR) | 0.9992 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9771 |
| Balanced Acc | 0.9771 |
| Macro-Sensitivity | 0.9771 |
| Macro-Specificity | 0.9975 |
| Macro-Precision | 0.9771 |
| AUC-PR (Macro) | 0.9955 |
| Cohen's Kappa | 0.9746 |
| MCC | 0.9746 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.27M |
| Pure-Inference FPS | 587.1 (1.70 ms/张) |
| End-to-End FPS | 444.2 |
| 测试样本数 | 10000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9860 | 0.9978 | 0.9801 | 0.9831 | 0.9984 | 0.9998 | 0.9983 | 1000 |
| automobile | 0.9890 | 0.9981 | 0.9831 | 0.9860 | 0.9988 | 0.9985 | 0.9969 | 1000 |
| bird | 0.9700 | 0.9978 | 0.9798 | 0.9749 | 0.9967 | 0.9993 | 0.9958 | 1000 |
| cat | 0.9450 | 0.9940 | 0.9459 | 0.9455 | 0.9939 | 0.9975 | 0.9845 | 1000 |
| deer | 0.9760 | 0.9980 | 0.9819 | 0.9789 | 0.9973 | 0.9997 | 0.9975 | 1000 |
| dog | 0.9590 | 0.9953 | 0.9580 | 0.9585 | 0.9954 | 0.9982 | 0.9899 | 1000 |
| frog | 0.9870 | 0.9980 | 0.9821 | 0.9845 | 0.9986 | 0.9998 | 0.9979 | 1000 |
| horse | 0.9890 | 0.9980 | 0.9821 | 0.9856 | 0.9988 | 0.9999 | 0.9994 | 1000 |
| ship | 0.9890 | 0.9987 | 0.9880 | 0.9885 | 0.9988 | 0.9999 | 0.9991 | 1000 |
| truck | 0.9810 | 0.9989 | 0.9899 | 0.9854 | 0.9979 | 0.9995 | 0.9958 | 1000 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=986 FP=20 TN=8980 FN=14
- **automobile**: TP=989 FP=17 TN=8983 FN=11
- **bird**: TP=970 FP=20 TN=8980 FN=30
- **cat**: TP=945 FP=54 TN=8946 FN=55
- **deer**: TP=976 FP=18 TN=8982 FN=24
- **dog**: TP=959 FP=42 TN=8958 FN=41
- **frog**: TP=987 FP=18 TN=8982 FN=13
- **horse**: TP=989 FP=18 TN=8982 FN=11
- **ship**: TP=989 FP=12 TN=8988 FN=11
- **truck**: TP=981 FP=10 TN=8990 FN=19

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 986 | 0 | 3 | 1 | 0 | 0 | 0 | 1 | 9 | 0 |
| automobile | 1 | 989 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 9 |
| bird | 5 | 0 | 970 | 7 | 6 | 3 | 8 | 1 | 0 | 0 |
| cat | 1 | 0 | 6 | 945 | 3 | 33 | 6 | 5 | 1 | 0 |
| deer | 0 | 0 | 6 | 3 | 976 | 2 | 3 | 10 | 0 | 0 |
| dog | 0 | 0 | 0 | 38 | 3 | 959 | 0 | 0 | 0 | 0 |
| frog | 2 | 0 | 4 | 3 | 2 | 1 | 987 | 1 | 0 | 0 |
| horse | 0 | 0 | 1 | 2 | 4 | 3 | 1 | 989 | 0 | 0 |
| ship | 9 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 989 | 1 |
| truck | 2 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 981 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9801    0.9860    0.9831      1000
  automobile     0.9831    0.9890    0.9860      1000
        bird     0.9798    0.9700    0.9749      1000
         cat     0.9459    0.9450    0.9455      1000
        deer     0.9819    0.9760    0.9789      1000
         dog     0.9580    0.9590    0.9585      1000
        frog     0.9821    0.9870    0.9845      1000
       horse     0.9821    0.9890    0.9856      1000
        ship     0.9880    0.9890    0.9885      1000
       truck     0.9899    0.9810    0.9854      1000

    accuracy                         0.9771     10000
   macro avg     0.9771    0.9771    0.9771     10000
weighted avg     0.9771    0.9771    0.9771     10000

```