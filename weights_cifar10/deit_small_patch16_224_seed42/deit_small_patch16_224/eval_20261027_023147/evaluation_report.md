# deit_small_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/deit_small_patch16_224_seed42/deit_small_patch16_224/model_best.pth`
- 测试集: `./datasets/cifar10/test`  (共 10000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9681 |
| Accuracy | 0.9681 |
| Macro-AUROC (OvR) | 0.9988 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9681 |
| Balanced Acc | 0.9681 |
| Macro-Sensitivity | 0.9681 |
| Macro-Specificity | 0.9965 |
| Macro-Precision | 0.9682 |
| AUC-PR (Macro) | 0.9927 |
| Cohen's Kappa | 0.9646 |
| MCC | 0.9646 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.25G |
| FLOPs | 8.50G |
| Params | 21.59M |
| Pure-Inference FPS | 971.7 (1.03 ms/张) |
| End-to-End FPS | 635.5 |
| 测试样本数 | 10000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9730 | 0.9979 | 0.9808 | 0.9769 | 0.9970 | 0.9996 | 0.9967 | 1000 |
| automobile | 0.9920 | 0.9964 | 0.9688 | 0.9802 | 0.9991 | 0.9993 | 0.9960 | 1000 |
| bird | 0.9610 | 0.9963 | 0.9668 | 0.9639 | 0.9957 | 0.9983 | 0.9922 | 1000 |
| cat | 0.9380 | 0.9912 | 0.9223 | 0.9301 | 0.9931 | 0.9963 | 0.9780 | 1000 |
| deer | 0.9670 | 0.9978 | 0.9797 | 0.9733 | 0.9963 | 0.9993 | 0.9942 | 1000 |
| dog | 0.9290 | 0.9940 | 0.9451 | 0.9370 | 0.9921 | 0.9974 | 0.9845 | 1000 |
| frog | 0.9840 | 0.9976 | 0.9781 | 0.9811 | 0.9982 | 0.9995 | 0.9963 | 1000 |
| horse | 0.9830 | 0.9973 | 0.9762 | 0.9796 | 0.9981 | 0.9998 | 0.9980 | 1000 |
| ship | 0.9870 | 0.9973 | 0.9763 | 0.9816 | 0.9986 | 0.9996 | 0.9976 | 1000 |
| truck | 0.9670 | 0.9987 | 0.9877 | 0.9773 | 0.9963 | 0.9991 | 0.9934 | 1000 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=973 FP=19 TN=8981 FN=27
- **automobile**: TP=992 FP=32 TN=8968 FN=8
- **bird**: TP=961 FP=33 TN=8967 FN=39
- **cat**: TP=938 FP=79 TN=8921 FN=62
- **deer**: TP=967 FP=20 TN=8980 FN=33
- **dog**: TP=929 FP=54 TN=8946 FN=71
- **frog**: TP=984 FP=22 TN=8978 FN=16
- **horse**: TP=983 FP=24 TN=8976 FN=17
- **ship**: TP=987 FP=24 TN=8976 FN=13
- **truck**: TP=967 FP=12 TN=8988 FN=33

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 973 | 3 | 5 | 2 | 0 | 0 | 0 | 0 | 16 | 1 |
| automobile | 2 | 992 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| bird | 4 | 0 | 961 | 14 | 7 | 4 | 5 | 4 | 1 | 0 |
| cat | 2 | 1 | 7 | 938 | 2 | 35 | 8 | 3 | 2 | 2 |
| deer | 0 | 0 | 6 | 5 | 967 | 7 | 4 | 10 | 1 | 0 |
| dog | 0 | 1 | 6 | 48 | 4 | 929 | 4 | 5 | 1 | 2 |
| frog | 1 | 0 | 4 | 6 | 3 | 1 | 984 | 1 | 0 | 0 |
| horse | 2 | 0 | 1 | 3 | 4 | 6 | 1 | 983 | 0 | 0 |
| ship | 7 | 0 | 3 | 0 | 0 | 1 | 0 | 1 | 987 | 1 |
| truck | 1 | 27 | 1 | 1 | 0 | 0 | 0 | 0 | 3 | 967 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9808    0.9730    0.9769      1000
  automobile     0.9688    0.9920    0.9802      1000
        bird     0.9668    0.9610    0.9639      1000
         cat     0.9223    0.9380    0.9301      1000
        deer     0.9797    0.9670    0.9733      1000
         dog     0.9451    0.9290    0.9370      1000
        frog     0.9781    0.9840    0.9811      1000
       horse     0.9762    0.9830    0.9796      1000
        ship     0.9763    0.9870    0.9816      1000
       truck     0.9877    0.9670    0.9773      1000

    accuracy                         0.9681     10000
   macro avg     0.9682    0.9681    0.9681     10000
weighted avg     0.9682    0.9681    0.9681     10000

```