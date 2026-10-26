# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_tiny_seed42/convnext_tiny/model_best.pth`
- 测试集: `./datasets/cifar10/val`  (共 5000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9798 |
| Accuracy | 0.9798 |
| Macro-AUROC (OvR) | 0.9997 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9798 |
| Balanced Acc | 0.9798 |
| Macro-Sensitivity | 0.9798 |
| Macro-Specificity | 0.9978 |
| Macro-Precision | 0.9799 |
| AUC-PR (Macro) | 0.9976 |
| Cohen's Kappa | 0.9776 |
| MCC | 0.9776 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.81M |
| Pure-Inference FPS | 603.5 (1.66 ms/张) |
| End-to-End FPS | 455.2 |
| 测试样本数 | 5000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9860 | 0.9980 | 0.9821 | 0.9840 | 0.9984 | 0.9999 | 0.9989 | 500 |
| automobile | 0.9860 | 0.9984 | 0.9860 | 0.9860 | 0.9984 | 0.9999 | 0.9994 | 500 |
| bird | 0.9860 | 0.9967 | 0.9705 | 0.9782 | 0.9984 | 0.9995 | 0.9972 | 500 |
| cat | 0.9660 | 0.9931 | 0.9397 | 0.9527 | 0.9962 | 0.9988 | 0.9923 | 500 |
| deer | 0.9800 | 0.9984 | 0.9859 | 0.9829 | 0.9978 | 0.9998 | 0.9981 | 500 |
| dog | 0.9520 | 0.9969 | 0.9714 | 0.9616 | 0.9947 | 0.9993 | 0.9945 | 500 |
| frog | 0.9800 | 0.9993 | 0.9939 | 0.9869 | 0.9978 | 0.9998 | 0.9987 | 500 |
| horse | 0.9840 | 0.9984 | 0.9860 | 0.9850 | 0.9982 | 0.9998 | 0.9984 | 500 |
| ship | 0.9900 | 0.9996 | 0.9960 | 0.9930 | 0.9989 | 1.0000 | 0.9998 | 500 |
| truck | 0.9880 | 0.9987 | 0.9880 | 0.9880 | 0.9987 | 0.9999 | 0.9990 | 500 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=493 FP=9 TN=4491 FN=7
- **automobile**: TP=493 FP=7 TN=4493 FN=7
- **bird**: TP=493 FP=15 TN=4485 FN=7
- **cat**: TP=483 FP=31 TN=4469 FN=17
- **deer**: TP=490 FP=7 TN=4493 FN=10
- **dog**: TP=476 FP=14 TN=4486 FN=24
- **frog**: TP=490 FP=3 TN=4497 FN=10
- **horse**: TP=492 FP=7 TN=4493 FN=8
- **ship**: TP=495 FP=2 TN=4498 FN=5
- **truck**: TP=494 FP=6 TN=4494 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 493 | 0 | 4 | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| automobile | 2 | 493 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| bird | 2 | 0 | 493 | 4 | 1 | 0 | 0 | 0 | 0 | 0 |
| cat | 0 | 0 | 3 | 483 | 0 | 12 | 2 | 0 | 0 | 0 |
| deer | 0 | 0 | 2 | 2 | 490 | 1 | 0 | 4 | 1 | 0 |
| dog | 0 | 0 | 1 | 19 | 1 | 476 | 0 | 3 | 0 | 0 |
| frog | 0 | 1 | 3 | 5 | 1 | 0 | 490 | 0 | 0 | 0 |
| horse | 0 | 0 | 2 | 1 | 4 | 1 | 0 | 492 | 0 | 0 |
| ship | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 495 | 0 |
| truck | 1 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 494 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9821    0.9860    0.9840       500
  automobile     0.9860    0.9860    0.9860       500
        bird     0.9705    0.9860    0.9782       500
         cat     0.9397    0.9660    0.9527       500
        deer     0.9859    0.9800    0.9829       500
         dog     0.9714    0.9520    0.9616       500
        frog     0.9939    0.9800    0.9869       500
       horse     0.9860    0.9840    0.9850       500
        ship     0.9960    0.9900    0.9930       500
       truck     0.9880    0.9880    0.9880       500

    accuracy                         0.9798      5000
   macro avg     0.9799    0.9798    0.9798      5000
weighted avg     0.9799    0.9798    0.9798      5000

```