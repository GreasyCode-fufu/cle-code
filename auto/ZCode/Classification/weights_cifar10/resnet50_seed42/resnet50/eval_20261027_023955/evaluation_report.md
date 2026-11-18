# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/resnet50_seed42/resnet50/model_best.pth`
- 测试集: `./datasets/cifar10/val`  (共 5000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9654 |
| Accuracy | 0.9654 |
| Macro-AUROC (OvR) | 0.9991 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9654 |
| Balanced Acc | 0.9654 |
| Macro-Sensitivity | 0.9654 |
| Macro-Specificity | 0.9962 |
| Macro-Precision | 0.9654 |
| AUC-PR (Macro) | 0.9941 |
| Cohen's Kappa | 0.9616 |
| MCC | 0.9616 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.53M |
| Pure-Inference FPS | 1224.4 (0.82 ms/张) |
| End-to-End FPS | 740.7 |
| 测试样本数 | 5000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9780 | 0.9969 | 0.9722 | 0.9751 | 0.9976 | 0.9998 | 0.9983 | 500 |
| automobile | 0.9760 | 0.9982 | 0.9839 | 0.9799 | 0.9973 | 0.9998 | 0.9982 | 500 |
| bird | 0.9580 | 0.9960 | 0.9638 | 0.9609 | 0.9953 | 0.9994 | 0.9953 | 500 |
| cat | 0.9300 | 0.9913 | 0.9226 | 0.9263 | 0.9922 | 0.9969 | 0.9794 | 500 |
| deer | 0.9680 | 0.9978 | 0.9798 | 0.9738 | 0.9964 | 0.9996 | 0.9966 | 500 |
| dog | 0.9280 | 0.9929 | 0.9355 | 0.9317 | 0.9920 | 0.9982 | 0.9854 | 500 |
| frog | 0.9820 | 0.9962 | 0.9665 | 0.9742 | 0.9980 | 0.9995 | 0.9963 | 500 |
| horse | 0.9800 | 0.9969 | 0.9722 | 0.9761 | 0.9978 | 0.9986 | 0.9950 | 500 |
| ship | 0.9760 | 0.9984 | 0.9859 | 0.9809 | 0.9973 | 0.9998 | 0.9986 | 500 |
| truck | 0.9780 | 0.9969 | 0.9722 | 0.9751 | 0.9976 | 0.9997 | 0.9976 | 500 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=489 FP=14 TN=4486 FN=11
- **automobile**: TP=488 FP=8 TN=4492 FN=12
- **bird**: TP=479 FP=18 TN=4482 FN=21
- **cat**: TP=465 FP=39 TN=4461 FN=35
- **deer**: TP=484 FP=10 TN=4490 FN=16
- **dog**: TP=464 FP=32 TN=4468 FN=36
- **frog**: TP=491 FP=17 TN=4483 FN=9
- **horse**: TP=490 FP=14 TN=4486 FN=10
- **ship**: TP=488 FP=7 TN=4493 FN=12
- **truck**: TP=489 FP=14 TN=4486 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 489 | 1 | 3 | 1 | 0 | 0 | 1 | 0 | 2 | 3 |
| automobile | 1 | 488 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 9 |
| bird | 6 | 0 | 479 | 7 | 2 | 1 | 4 | 1 | 0 | 0 |
| cat | 0 | 0 | 5 | 465 | 1 | 22 | 7 | 0 | 0 | 0 |
| deer | 0 | 0 | 1 | 1 | 484 | 4 | 2 | 8 | 0 | 0 |
| dog | 2 | 0 | 3 | 22 | 3 | 464 | 1 | 5 | 0 | 0 |
| frog | 1 | 0 | 3 | 2 | 1 | 1 | 491 | 0 | 1 | 0 |
| horse | 0 | 0 | 3 | 3 | 2 | 2 | 0 | 490 | 0 | 0 |
| ship | 3 | 3 | 0 | 2 | 0 | 2 | 0 | 0 | 488 | 2 |
| truck | 1 | 4 | 0 | 0 | 1 | 0 | 2 | 0 | 3 | 489 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9722    0.9780    0.9751       500
  automobile     0.9839    0.9760    0.9799       500
        bird     0.9638    0.9580    0.9609       500
         cat     0.9226    0.9300    0.9263       500
        deer     0.9798    0.9680    0.9738       500
         dog     0.9355    0.9280    0.9317       500
        frog     0.9665    0.9820    0.9742       500
       horse     0.9722    0.9800    0.9761       500
        ship     0.9859    0.9760    0.9809       500
       truck     0.9722    0.9780    0.9751       500

    accuracy                         0.9654      5000
   macro avg     0.9654    0.9654    0.9654      5000
weighted avg     0.9654    0.9654    0.9654      5000

```