# convnext_v2b 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_v2b_seed42/convnext_v2b/model_best.pth`
- 测试集: `./datasets/cifar10/val`  (共 5000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9824 |
| Accuracy | 0.9824 |
| Macro-AUROC (OvR) | 0.9997 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9824 |
| Balanced Acc | 0.9824 |
| Macro-Sensitivity | 0.9824 |
| Macro-Specificity | 0.9980 |
| Macro-Precision | 0.9825 |
| AUC-PR (Macro) | 0.9975 |
| Cohen's Kappa | 0.9804 |
| MCC | 0.9804 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.93G |
| Params | 28.22M |
| Pure-Inference FPS | 1605.1 (0.62 ms/张) |
| End-to-End FPS | 880.9 |
| 测试样本数 | 5000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9880 | 0.9982 | 0.9841 | 0.9860 | 0.9987 | 0.9999 | 0.9992 | 500 |
| automobile | 0.9900 | 0.9996 | 0.9960 | 0.9930 | 0.9989 | 0.9999 | 0.9995 | 500 |
| bird | 0.9840 | 0.9980 | 0.9820 | 0.9830 | 0.9982 | 0.9997 | 0.9978 | 500 |
| cat | 0.9600 | 0.9942 | 0.9486 | 0.9543 | 0.9955 | 0.9989 | 0.9906 | 500 |
| deer | 0.9760 | 0.9991 | 0.9919 | 0.9839 | 0.9973 | 0.9998 | 0.9982 | 500 |
| dog | 0.9640 | 0.9962 | 0.9659 | 0.9650 | 0.9960 | 0.9993 | 0.9945 | 500 |
| frog | 0.9860 | 0.9993 | 0.9940 | 0.9900 | 0.9984 | 0.9996 | 0.9979 | 500 |
| horse | 0.9960 | 0.9978 | 0.9803 | 0.9881 | 0.9996 | 0.9998 | 0.9988 | 500 |
| ship | 0.9900 | 0.9996 | 0.9960 | 0.9930 | 0.9989 | 1.0000 | 0.9999 | 500 |
| truck | 0.9900 | 0.9984 | 0.9861 | 0.9880 | 0.9989 | 0.9999 | 0.9988 | 500 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=494 FP=8 TN=4492 FN=6
- **automobile**: TP=495 FP=2 TN=4498 FN=5
- **bird**: TP=492 FP=9 TN=4491 FN=8
- **cat**: TP=480 FP=26 TN=4474 FN=20
- **deer**: TP=488 FP=4 TN=4496 FN=12
- **dog**: TP=482 FP=17 TN=4483 FN=18
- **frog**: TP=493 FP=3 TN=4497 FN=7
- **horse**: TP=498 FP=10 TN=4490 FN=2
- **ship**: TP=495 FP=2 TN=4498 FN=5
- **truck**: TP=495 FP=7 TN=4493 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 494 | 0 | 4 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| automobile | 1 | 495 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| bird | 3 | 0 | 492 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| cat | 0 | 0 | 4 | 480 | 0 | 15 | 1 | 0 | 0 | 0 |
| deer | 0 | 0 | 1 | 3 | 488 | 1 | 0 | 7 | 0 | 0 |
| dog | 0 | 0 | 0 | 17 | 0 | 482 | 0 | 1 | 0 | 0 |
| frog | 1 | 0 | 0 | 4 | 2 | 0 | 493 | 0 | 0 | 0 |
| horse | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 498 | 0 | 0 |
| ship | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 495 | 2 |
| truck | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 495 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9841    0.9880    0.9860       500
  automobile     0.9960    0.9900    0.9930       500
        bird     0.9820    0.9840    0.9830       500
         cat     0.9486    0.9600    0.9543       500
        deer     0.9919    0.9760    0.9839       500
         dog     0.9659    0.9640    0.9650       500
        frog     0.9940    0.9860    0.9900       500
       horse     0.9803    0.9960    0.9881       500
        ship     0.9960    0.9900    0.9930       500
       truck     0.9861    0.9900    0.9880       500

    accuracy                         0.9824      5000
   macro avg     0.9825    0.9824    0.9824      5000
weighted avg     0.9825    0.9824    0.9824      5000

```