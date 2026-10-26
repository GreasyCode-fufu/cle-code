# convnext_v2b 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_v2b_seed42/convnext_v2b/model_best.pth`
- 测试集: `./datasets/cifar10/test`  (共 10000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9756 |
| Accuracy | 0.9756 |
| Macro-AUROC (OvR) | 0.9993 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9756 |
| Balanced Acc | 0.9756 |
| Macro-Sensitivity | 0.9756 |
| Macro-Specificity | 0.9973 |
| Macro-Precision | 0.9756 |
| AUC-PR (Macro) | 0.9954 |
| Cohen's Kappa | 0.9729 |
| MCC | 0.9729 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.93G |
| Params | 28.22M |
| Pure-Inference FPS | 1616.4 (0.62 ms/张) |
| End-to-End FPS | 884.6 |
| 测试样本数 | 10000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9830 | 0.9973 | 0.9762 | 0.9796 | 0.9981 | 0.9997 | 0.9979 | 1000 |
| automobile | 0.9890 | 0.9978 | 0.9802 | 0.9846 | 0.9988 | 0.9994 | 0.9968 | 1000 |
| bird | 0.9720 | 0.9978 | 0.9798 | 0.9759 | 0.9969 | 0.9996 | 0.9970 | 1000 |
| cat | 0.9420 | 0.9941 | 0.9467 | 0.9444 | 0.9936 | 0.9974 | 0.9827 | 1000 |
| deer | 0.9730 | 0.9977 | 0.9789 | 0.9759 | 0.9970 | 0.9996 | 0.9968 | 1000 |
| dog | 0.9580 | 0.9949 | 0.9542 | 0.9561 | 0.9953 | 0.9981 | 0.9899 | 1000 |
| frog | 0.9850 | 0.9980 | 0.9821 | 0.9835 | 0.9983 | 0.9999 | 0.9986 | 1000 |
| horse | 0.9890 | 0.9984 | 0.9860 | 0.9875 | 0.9988 | 0.9999 | 0.9990 | 1000 |
| ship | 0.9890 | 0.9984 | 0.9860 | 0.9875 | 0.9988 | 0.9999 | 0.9994 | 1000 |
| truck | 0.9760 | 0.9984 | 0.9859 | 0.9809 | 0.9973 | 0.9995 | 0.9961 | 1000 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=983 FP=24 TN=8976 FN=17
- **automobile**: TP=989 FP=20 TN=8980 FN=11
- **bird**: TP=972 FP=20 TN=8980 FN=28
- **cat**: TP=942 FP=53 TN=8947 FN=58
- **deer**: TP=973 FP=21 TN=8979 FN=27
- **dog**: TP=958 FP=46 TN=8954 FN=42
- **frog**: TP=985 FP=18 TN=8982 FN=15
- **horse**: TP=989 FP=14 TN=8986 FN=11
- **ship**: TP=989 FP=14 TN=8986 FN=11
- **truck**: TP=976 FP=14 TN=8986 FN=24

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 983 | 0 | 2 | 2 | 1 | 0 | 0 | 2 | 10 | 0 |
| automobile | 2 | 989 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |
| bird | 4 | 0 | 972 | 6 | 5 | 5 | 7 | 0 | 1 | 0 |
| cat | 1 | 0 | 4 | 942 | 6 | 35 | 5 | 4 | 1 | 2 |
| deer | 0 | 0 | 8 | 3 | 973 | 3 | 4 | 8 | 0 | 1 |
| dog | 0 | 0 | 1 | 35 | 4 | 958 | 2 | 0 | 0 | 0 |
| frog | 3 | 1 | 5 | 5 | 0 | 1 | 985 | 0 | 0 | 0 |
| horse | 2 | 0 | 0 | 2 | 5 | 2 | 0 | 989 | 0 | 0 |
| ship | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 989 | 2 |
| truck | 3 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 976 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9762    0.9830    0.9796      1000
  automobile     0.9802    0.9890    0.9846      1000
        bird     0.9798    0.9720    0.9759      1000
         cat     0.9467    0.9420    0.9444      1000
        deer     0.9789    0.9730    0.9759      1000
         dog     0.9542    0.9580    0.9561      1000
        frog     0.9821    0.9850    0.9835      1000
       horse     0.9860    0.9890    0.9875      1000
        ship     0.9860    0.9890    0.9875      1000
       truck     0.9859    0.9760    0.9809      1000

    accuracy                         0.9756     10000
   macro avg     0.9756    0.9756    0.9756     10000
weighted avg     0.9756    0.9756    0.9756     10000

```