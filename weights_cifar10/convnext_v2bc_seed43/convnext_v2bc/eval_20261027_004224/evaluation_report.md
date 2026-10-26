# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_v2bc_seed43/convnext_v2bc/model_best.pth`
- 测试集: `./datasets/cifar10/val`  (共 5000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9798 |
| Accuracy | 0.9798 |
| Macro-AUROC (OvR) | 0.9995 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9798 |
| Balanced Acc | 0.9798 |
| Macro-Sensitivity | 0.9798 |
| Macro-Specificity | 0.9978 |
| Macro-Precision | 0.9799 |
| AUC-PR (Macro) | 0.9969 |
| Cohen's Kappa | 0.9776 |
| MCC | 0.9776 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.68M |
| Pure-Inference FPS | 596.7 (1.68 ms/张) |
| End-to-End FPS | 454.8 |
| 测试样本数 | 5000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9900 | 0.9980 | 0.9821 | 0.9861 | 0.9989 | 0.9999 | 0.9992 | 500 |
| automobile | 0.9920 | 0.9984 | 0.9861 | 0.9890 | 0.9991 | 0.9997 | 0.9980 | 500 |
| bird | 0.9820 | 0.9971 | 0.9742 | 0.9781 | 0.9980 | 0.9997 | 0.9973 | 500 |
| cat | 0.9680 | 0.9942 | 0.9490 | 0.9584 | 0.9964 | 0.9974 | 0.9904 | 500 |
| deer | 0.9760 | 0.9991 | 0.9919 | 0.9839 | 0.9973 | 0.9997 | 0.9979 | 500 |
| dog | 0.9560 | 0.9962 | 0.9657 | 0.9608 | 0.9951 | 0.9989 | 0.9916 | 500 |
| frog | 0.9720 | 0.9989 | 0.9898 | 0.9808 | 0.9969 | 0.9998 | 0.9985 | 500 |
| horse | 0.9880 | 0.9980 | 0.9821 | 0.9850 | 0.9987 | 0.9998 | 0.9986 | 500 |
| ship | 0.9860 | 0.9989 | 0.9900 | 0.9880 | 0.9984 | 0.9999 | 0.9993 | 500 |
| truck | 0.9880 | 0.9987 | 0.9880 | 0.9880 | 0.9987 | 0.9998 | 0.9986 | 500 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=495 FP=9 TN=4491 FN=5
- **automobile**: TP=496 FP=7 TN=4493 FN=4
- **bird**: TP=491 FP=13 TN=4487 FN=9
- **cat**: TP=484 FP=26 TN=4474 FN=16
- **deer**: TP=488 FP=4 TN=4496 FN=12
- **dog**: TP=478 FP=17 TN=4483 FN=22
- **frog**: TP=486 FP=5 TN=4495 FN=14
- **horse**: TP=494 FP=9 TN=4491 FN=6
- **ship**: TP=493 FP=5 TN=4495 FN=7
- **truck**: TP=494 FP=6 TN=4494 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 495 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 2 |
| automobile | 0 | 496 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| bird | 3 | 0 | 491 | 1 | 1 | 0 | 1 | 0 | 3 | 0 |
| cat | 0 | 1 | 3 | 484 | 0 | 10 | 1 | 1 | 0 | 0 |
| deer | 0 | 0 | 3 | 0 | 488 | 3 | 1 | 5 | 0 | 0 |
| dog | 0 | 0 | 0 | 18 | 0 | 478 | 2 | 2 | 0 | 0 |
| frog | 1 | 0 | 4 | 6 | 1 | 1 | 486 | 0 | 1 | 0 |
| horse | 0 | 0 | 1 | 1 | 2 | 2 | 0 | 494 | 0 | 0 |
| ship | 4 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 493 | 0 |
| truck | 1 | 4 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 494 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9821    0.9900    0.9861       500
  automobile     0.9861    0.9920    0.9890       500
        bird     0.9742    0.9820    0.9781       500
         cat     0.9490    0.9680    0.9584       500
        deer     0.9919    0.9760    0.9839       500
         dog     0.9657    0.9560    0.9608       500
        frog     0.9898    0.9720    0.9808       500
       horse     0.9821    0.9880    0.9850       500
        ship     0.9900    0.9860    0.9880       500
       truck     0.9880    0.9880    0.9880       500

    accuracy                         0.9798      5000
   macro avg     0.9799    0.9798    0.9798      5000
weighted avg     0.9799    0.9798    0.9798      5000

```