# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_v2c_seed43/convnext_v2c/model_best.pth`
- 测试集: `./datasets/cifar10/val`  (共 5000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9820 |
| Accuracy | 0.9820 |
| Macro-AUROC (OvR) | 0.9996 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9820 |
| Balanced Acc | 0.9820 |
| Macro-Sensitivity | 0.9820 |
| Macro-Specificity | 0.9980 |
| Macro-Precision | 0.9820 |
| AUC-PR (Macro) | 0.9971 |
| Cohen's Kappa | 0.9800 |
| MCC | 0.9800 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.27M |
| Pure-Inference FPS | 648.3 (1.54 ms/张) |
| End-to-End FPS | 480.7 |
| 测试样本数 | 5000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9940 | 0.9980 | 0.9822 | 0.9881 | 0.9993 | 0.9999 | 0.9995 | 500 |
| automobile | 0.9940 | 0.9987 | 0.9881 | 0.9910 | 0.9993 | 0.9999 | 0.9991 | 500 |
| bird | 0.9860 | 0.9987 | 0.9880 | 0.9870 | 0.9984 | 0.9997 | 0.9981 | 500 |
| cat | 0.9600 | 0.9953 | 0.9581 | 0.9590 | 0.9956 | 0.9988 | 0.9906 | 500 |
| deer | 0.9820 | 0.9989 | 0.9899 | 0.9859 | 0.9980 | 0.9995 | 0.9975 | 500 |
| dog | 0.9580 | 0.9964 | 0.9677 | 0.9628 | 0.9953 | 0.9986 | 0.9914 | 500 |
| frog | 0.9800 | 0.9984 | 0.9859 | 0.9829 | 0.9978 | 0.9999 | 0.9988 | 500 |
| horse | 0.9920 | 0.9976 | 0.9783 | 0.9851 | 0.9991 | 0.9998 | 0.9984 | 500 |
| ship | 0.9840 | 0.9993 | 0.9939 | 0.9889 | 0.9982 | 0.9998 | 0.9986 | 500 |
| truck | 0.9900 | 0.9987 | 0.9880 | 0.9890 | 0.9989 | 0.9999 | 0.9992 | 500 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=497 FP=9 TN=4491 FN=3
- **automobile**: TP=497 FP=6 TN=4494 FN=3
- **bird**: TP=493 FP=6 TN=4494 FN=7
- **cat**: TP=480 FP=21 TN=4479 FN=20
- **deer**: TP=491 FP=5 TN=4495 FN=9
- **dog**: TP=479 FP=16 TN=4484 FN=21
- **frog**: TP=490 FP=7 TN=4493 FN=10
- **horse**: TP=496 FP=11 TN=4489 FN=4
- **ship**: TP=492 FP=3 TN=4497 FN=8
- **truck**: TP=495 FP=6 TN=4494 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 497 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| automobile | 0 | 497 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| bird | 3 | 0 | 493 | 0 | 1 | 0 | 1 | 1 | 1 | 0 |
| cat | 1 | 0 | 4 | 480 | 0 | 11 | 2 | 2 | 0 | 0 |
| deer | 0 | 0 | 1 | 0 | 491 | 3 | 1 | 4 | 0 | 0 |
| dog | 0 | 0 | 0 | 15 | 0 | 479 | 2 | 4 | 0 | 0 |
| frog | 1 | 0 | 0 | 5 | 3 | 0 | 490 | 0 | 1 | 0 |
| horse | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 496 | 0 | 0 |
| ship | 4 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 492 | 1 |
| truck | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 495 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9822    0.9940    0.9881       500
  automobile     0.9881    0.9940    0.9910       500
        bird     0.9880    0.9860    0.9870       500
         cat     0.9581    0.9600    0.9590       500
        deer     0.9899    0.9820    0.9859       500
         dog     0.9677    0.9580    0.9628       500
        frog     0.9859    0.9800    0.9829       500
       horse     0.9783    0.9920    0.9851       500
        ship     0.9939    0.9840    0.9889       500
       truck     0.9880    0.9900    0.9890       500

    accuracy                         0.9820      5000
   macro avg     0.9820    0.9820    0.9820      5000
weighted avg     0.9820    0.9820    0.9820      5000

```