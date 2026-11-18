# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cifar10/convnext_v2bc_seed42/convnext_v2bc/model_best.pth`
- 测试集: `./datasets/cifar10/val`  (共 5000 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9782 |
| Accuracy | 0.9782 |
| Macro-AUROC (OvR) | 0.9996 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9782 |
| Balanced Acc | 0.9782 |
| Macro-Sensitivity | 0.9782 |
| Macro-Specificity | 0.9976 |
| Macro-Precision | 0.9783 |
| AUC-PR (Macro) | 0.9973 |
| Cohen's Kappa | 0.9758 |
| MCC | 0.9758 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.68M |
| Pure-Inference FPS | 594.9 (1.68 ms/张) |
| End-to-End FPS | 452.7 |
| 测试样本数 | 5000 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.9900 | 0.9969 | 0.9725 | 0.9812 | 0.9989 | 0.9999 | 0.9991 | 500 |
| automobile | 0.9940 | 0.9987 | 0.9881 | 0.9910 | 0.9993 | 0.9997 | 0.9980 | 500 |
| bird | 0.9680 | 0.9989 | 0.9898 | 0.9788 | 0.9965 | 0.9998 | 0.9983 | 500 |
| cat | 0.9580 | 0.9942 | 0.9485 | 0.9532 | 0.9953 | 0.9988 | 0.9919 | 500 |
| deer | 0.9780 | 0.9978 | 0.9800 | 0.9790 | 0.9976 | 0.9998 | 0.9980 | 500 |
| dog | 0.9520 | 0.9964 | 0.9675 | 0.9597 | 0.9947 | 0.9989 | 0.9925 | 500 |
| frog | 0.9860 | 0.9984 | 0.9860 | 0.9860 | 0.9984 | 0.9999 | 0.9992 | 500 |
| horse | 0.9860 | 0.9971 | 0.9743 | 0.9801 | 0.9984 | 0.9998 | 0.9986 | 500 |
| ship | 0.9860 | 0.9984 | 0.9860 | 0.9860 | 0.9984 | 0.9999 | 0.9994 | 500 |
| truck | 0.9840 | 0.9989 | 0.9899 | 0.9870 | 0.9982 | 0.9998 | 0.9981 | 500 |

### 每类 TP/FP/TN/FN

- **airplane**: TP=495 FP=14 TN=4486 FN=5
- **automobile**: TP=497 FP=6 TN=4494 FN=3
- **bird**: TP=484 FP=5 TN=4495 FN=16
- **cat**: TP=479 FP=26 TN=4474 FN=21
- **deer**: TP=489 FP=10 TN=4490 FN=11
- **dog**: TP=476 FP=16 TN=4484 FN=24
- **frog**: TP=493 FP=7 TN=4493 FN=7
- **horse**: TP=493 FP=13 TN=4487 FN=7
- **ship**: TP=493 FP=7 TN=4493 FN=7
- **truck**: TP=492 FP=5 TN=4495 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| airplane | 495 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| automobile | 0 | 497 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| bird | 6 | 0 | 484 | 2 | 2 | 0 | 2 | 1 | 3 | 0 |
| cat | 2 | 1 | 2 | 479 | 2 | 12 | 1 | 1 | 0 | 0 |
| deer | 0 | 0 | 0 | 2 | 489 | 3 | 0 | 6 | 0 | 0 |
| dog | 0 | 0 | 0 | 16 | 1 | 476 | 3 | 4 | 0 | 0 |
| frog | 1 | 0 | 0 | 3 | 2 | 0 | 493 | 0 | 1 | 0 |
| horse | 0 | 0 | 1 | 3 | 2 | 1 | 0 | 493 | 0 | 0 |
| ship | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 493 | 1 |
| truck | 1 | 4 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 492 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    airplane     0.9725    0.9900    0.9812       500
  automobile     0.9881    0.9940    0.9910       500
        bird     0.9898    0.9680    0.9788       500
         cat     0.9485    0.9580    0.9532       500
        deer     0.9800    0.9780    0.9790       500
         dog     0.9675    0.9520    0.9597       500
        frog     0.9860    0.9860    0.9860       500
       horse     0.9743    0.9860    0.9801       500
        ship     0.9860    0.9860    0.9860       500
       truck     0.9899    0.9840    0.9870       500

    accuracy                         0.9782      5000
   macro avg     0.9783    0.9782    0.9782      5000
weighted avg     0.9783    0.9782    0.9782      5000

```