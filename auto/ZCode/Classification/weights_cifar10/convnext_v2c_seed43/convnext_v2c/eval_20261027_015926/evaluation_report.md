# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_v2c_seed43/convnext_v2c/model_best.pth`
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
| AUC-PR (Macro) | 0.9954 |
| Cohen's Kappa | 0.9722 |
| MCC | 0.9722 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.27M |
| Pure-Inference FPS | 589.8 (1.70 ms/张) |
| End-to-End FPS | 443.9 |
| 测试样本数 | 10000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9850 | 0.9976 | 0.9782 | 0.9816 | 0.9983 | 0.9998 | 0.9981 | 1000 |
| automobile | 0.9880 | 0.9972 | 0.9753 | 0.9816 | 0.9987 | 0.9991 | 0.9969 | 1000 |
| bird | 0.9680 | 0.9977 | 0.9788 | 0.9734 | 0.9964 | 0.9994 | 0.9959 | 1000 |
| cat | 0.9370 | 0.9944 | 0.9493 | 0.9431 | 0.9930 | 0.9977 | 0.9840 | 1000 |
| deer | 0.9790 | 0.9977 | 0.9790 | 0.9790 | 0.9977 | 0.9997 | 0.9975 | 1000 |
| dog | 0.9580 | 0.9943 | 0.9495 | 0.9537 | 0.9953 | 0.9984 | 0.9895 | 1000 |
| frog | 0.9920 | 0.9977 | 0.9793 | 0.9856 | 0.9991 | 0.9998 | 0.9982 | 1000 |
| horse | 0.9820 | 0.9989 | 0.9899 | 0.9859 | 0.9980 | 0.9999 | 0.9990 | 1000 |
| ship | 0.9890 | 0.9982 | 0.9841 | 0.9865 | 0.9988 | 0.9999 | 0.9989 | 1000 |
| truck | 0.9720 | 0.9986 | 0.9868 | 0.9793 | 0.9969 | 0.9995 | 0.9960 | 1000 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=985 FP=22 TN=8978 FN=15
- **automobile**: TP=988 FP=25 TN=8975 FN=12
- **bird**: TP=968 FP=21 TN=8979 FN=32
- **cat**: TP=937 FP=50 TN=8950 FN=63
- **deer**: TP=979 FP=21 TN=8979 FN=21
- **dog**: TP=958 FP=51 TN=8949 FN=42
- **frog**: TP=992 FP=21 TN=8979 FN=8
- **horse**: TP=982 FP=10 TN=8990 FN=18
- **ship**: TP=989 FP=16 TN=8984 FN=11
- **truck**: TP=972 FP=13 TN=8987 FN=28

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 985 | 0 | 2 | 1 | 0 | 0 | 0 | 2 | 10 | 0 |
| automobile | 2 | 988 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 9 |
| bird | 3 | 0 | 968 | 7 | 4 | 3 | 11 | 2 | 2 | 0 |
| cat | 1 | 0 | 7 | 937 | 5 | 38 | 8 | 2 | 1 | 1 |
| deer | 0 | 0 | 6 | 6 | 979 | 4 | 1 | 3 | 1 | 0 |
| dog | 1 | 0 | 2 | 35 | 3 | 958 | 1 | 0 | 0 | 0 |
| frog | 3 | 0 | 1 | 0 | 2 | 1 | 992 | 1 | 0 | 0 |
| horse | 2 | 0 | 3 | 1 | 7 | 5 | 0 | 982 | 0 | 0 |
| ship | 7 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 989 | 3 |
| truck | 3 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 972 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9782    0.9850    0.9816      1000
  automobile     0.9753    0.9880    0.9816      1000
        bird     0.9788    0.9680    0.9734      1000
         cat     0.9493    0.9370    0.9431      1000
        deer     0.9790    0.9790    0.9790      1000
         dog     0.9495    0.9580    0.9537      1000
        frog     0.9793    0.9920    0.9856      1000
       horse     0.9899    0.9820    0.9859      1000
        ship     0.9841    0.9890    0.9865      1000
       truck     0.9868    0.9720    0.9793      1000

    accuracy                         0.9750     10000
   macro avg     0.9750    0.9750    0.9750     10000
weighted avg     0.9750    0.9750    0.9750     10000

```