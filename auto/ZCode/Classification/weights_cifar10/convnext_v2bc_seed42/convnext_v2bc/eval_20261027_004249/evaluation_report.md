# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_v2bc_seed42/convnext_v2bc/model_best.pth`
- 测试集: `./datasets/cifar10/test`  (共 10000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9737 |
| Accuracy | 0.9737 |
| Macro-AUROC (OvR) | 0.9992 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9737 |
| Balanced Acc | 0.9737 |
| Macro-Sensitivity | 0.9737 |
| Macro-Specificity | 0.9971 |
| Macro-Precision | 0.9737 |
| AUC-PR (Macro) | 0.9951 |
| Cohen's Kappa | 0.9708 |
| MCC | 0.9708 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.68M |
| Pure-Inference FPS | 596.1 (1.68 ms/张) |
| End-to-End FPS | 454.0 |
| 测试样本数 | 10000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9740 | 0.9976 | 0.9779 | 0.9760 | 0.9971 | 0.9997 | 0.9977 | 1000 |
| automobile | 0.9870 | 0.9973 | 0.9763 | 0.9816 | 0.9986 | 0.9993 | 0.9972 | 1000 |
| bird | 0.9660 | 0.9979 | 0.9807 | 0.9733 | 0.9962 | 0.9992 | 0.9954 | 1000 |
| cat | 0.9340 | 0.9939 | 0.9444 | 0.9392 | 0.9927 | 0.9975 | 0.9832 | 1000 |
| deer | 0.9810 | 0.9969 | 0.9722 | 0.9766 | 0.9979 | 0.9997 | 0.9968 | 1000 |
| dog | 0.9560 | 0.9944 | 0.9503 | 0.9531 | 0.9951 | 0.9979 | 0.9884 | 1000 |
| frog | 0.9890 | 0.9982 | 0.9841 | 0.9865 | 0.9988 | 0.9999 | 0.9986 | 1000 |
| horse | 0.9850 | 0.9989 | 0.9899 | 0.9875 | 0.9983 | 0.9999 | 0.9991 | 1000 |
| ship | 0.9870 | 0.9976 | 0.9782 | 0.9826 | 0.9986 | 0.9998 | 0.9986 | 1000 |
| truck | 0.9780 | 0.9981 | 0.9829 | 0.9805 | 0.9976 | 0.9996 | 0.9964 | 1000 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=974 FP=22 TN=8978 FN=26
- **automobile**: TP=987 FP=24 TN=8976 FN=13
- **bird**: TP=966 FP=19 TN=8981 FN=34
- **cat**: TP=934 FP=55 TN=8945 FN=66
- **deer**: TP=981 FP=28 TN=8972 FN=19
- **dog**: TP=956 FP=50 TN=8950 FN=44
- **frog**: TP=989 FP=16 TN=8984 FN=11
- **horse**: TP=985 FP=10 TN=8990 FN=15
- **ship**: TP=987 FP=22 TN=8978 FN=13
- **truck**: TP=978 FP=17 TN=8983 FN=22

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 974 | 3 | 2 | 2 | 0 | 0 | 0 | 0 | 16 | 3 |
| automobile | 2 | 987 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 10 |
| bird | 4 | 0 | 966 | 7 | 12 | 1 | 7 | 1 | 2 | 0 |
| cat | 1 | 1 | 6 | 934 | 4 | 44 | 6 | 2 | 1 | 1 |
| deer | 0 | 0 | 5 | 4 | 981 | 3 | 2 | 4 | 1 | 0 |
| dog | 0 | 0 | 0 | 37 | 4 | 956 | 1 | 2 | 0 | 0 |
| frog | 2 | 0 | 4 | 2 | 1 | 1 | 989 | 1 | 0 | 0 |
| horse | 2 | 0 | 2 | 3 | 7 | 1 | 0 | 985 | 0 | 0 |
| ship | 9 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 987 | 3 |
| truck | 2 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 978 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9779    0.9740    0.9760      1000
  automobile     0.9763    0.9870    0.9816      1000
        bird     0.9807    0.9660    0.9733      1000
         cat     0.9444    0.9340    0.9392      1000
        deer     0.9722    0.9810    0.9766      1000
         dog     0.9503    0.9560    0.9531      1000
        frog     0.9841    0.9890    0.9865      1000
       horse     0.9899    0.9850    0.9875      1000
        ship     0.9782    0.9870    0.9826      1000
       truck     0.9829    0.9780    0.9805      1000

    accuracy                         0.9737     10000
   macro avg     0.9737    0.9737    0.9737     10000
weighted avg     0.9737    0.9737    0.9737     10000

```