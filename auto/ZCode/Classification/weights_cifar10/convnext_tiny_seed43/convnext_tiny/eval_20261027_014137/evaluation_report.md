# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_tiny_seed43/convnext_tiny/model_best.pth`
- 测试集: `./datasets/cifar10/test`  (共 10000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9759 |
| Accuracy | 0.9759 |
| Macro-AUROC (OvR) | 0.9994 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9759 |
| Balanced Acc | 0.9759 |
| Macro-Sensitivity | 0.9759 |
| Macro-Specificity | 0.9973 |
| Macro-Precision | 0.9759 |
| AUC-PR (Macro) | 0.9958 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.81M |
| Pure-Inference FPS | 619.6 (1.61 ms/张) |
| End-to-End FPS | 466.4 |
| 测试样本数 | 10000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9780 | 0.9989 | 0.9899 | 0.9839 | 0.9976 | 0.9999 | 0.9987 | 1000 |
| automobile | 0.9890 | 0.9982 | 0.9841 | 0.9865 | 0.9988 | 0.9996 | 0.9977 | 1000 |
| bird | 0.9770 | 0.9968 | 0.9712 | 0.9741 | 0.9974 | 0.9995 | 0.9963 | 1000 |
| cat | 0.9390 | 0.9944 | 0.9494 | 0.9442 | 0.9932 | 0.9976 | 0.9844 | 1000 |
| deer | 0.9800 | 0.9962 | 0.9665 | 0.9732 | 0.9978 | 0.9996 | 0.9972 | 1000 |
| dog | 0.9510 | 0.9953 | 0.9577 | 0.9543 | 0.9946 | 0.9985 | 0.9898 | 1000 |
| frog | 0.9910 | 0.9984 | 0.9861 | 0.9885 | 0.9990 | 0.9998 | 0.9979 | 1000 |
| horse | 0.9830 | 0.9981 | 0.9830 | 0.9830 | 0.9981 | 0.9998 | 0.9985 | 1000 |
| ship | 0.9900 | 0.9984 | 0.9861 | 0.9880 | 0.9989 | 0.9999 | 0.9993 | 1000 |
| truck | 0.9810 | 0.9983 | 0.9849 | 0.9830 | 0.9979 | 0.9997 | 0.9978 | 1000 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=978 FP=10 TN=8990 FN=22
- **automobile**: TP=989 FP=16 TN=8984 FN=11
- **bird**: TP=977 FP=29 TN=8971 FN=23
- **cat**: TP=939 FP=50 TN=8950 FN=61
- **deer**: TP=980 FP=34 TN=8966 FN=20
- **dog**: TP=951 FP=42 TN=8958 FN=49
- **frog**: TP=991 FP=14 TN=8986 FN=9
- **horse**: TP=983 FP=17 TN=8983 FN=17
- **ship**: TP=990 FP=14 TN=8986 FN=10
- **truck**: TP=981 FP=15 TN=8985 FN=19

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 978 | 0 | 7 | 1 | 1 | 1 | 0 | 1 | 11 | 0 |
| automobile | 1 | 989 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |
| bird | 0 | 0 | 977 | 9 | 8 | 2 | 4 | 0 | 0 | 0 |
| cat | 0 | 0 | 6 | 939 | 9 | 34 | 6 | 4 | 1 | 1 |
| deer | 0 | 0 | 8 | 1 | 980 | 2 | 2 | 7 | 0 | 0 |
| dog | 0 | 0 | 1 | 37 | 4 | 951 | 1 | 5 | 0 | 1 |
| frog | 2 | 0 | 2 | 1 | 2 | 1 | 991 | 0 | 0 | 1 |
| horse | 0 | 0 | 4 | 1 | 10 | 2 | 0 | 983 | 0 | 0 |
| ship | 6 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 990 | 2 |
| truck | 1 | 15 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 981 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9899    0.9780    0.9839      1000
  automobile     0.9841    0.9890    0.9865      1000
        bird     0.9712    0.9770    0.9741      1000
         cat     0.9494    0.9390    0.9442      1000
        deer     0.9665    0.9800    0.9732      1000
         dog     0.9577    0.9510    0.9543      1000
        frog     0.9861    0.9910    0.9885      1000
       horse     0.9830    0.9830    0.9830      1000
        ship     0.9861    0.9900    0.9880      1000
       truck     0.9849    0.9810    0.9830      1000

    accuracy                         0.9759     10000
   macro avg     0.9759    0.9759    0.9759     10000
weighted avg     0.9759    0.9759    0.9759     10000

```