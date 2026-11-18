# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/resnet50_seed44/resnet50/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8425 |
| Accuracy | 0.8739 |
| Macro-AUROC (OvR) | 0.9786 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8747 |
| Balanced Acc | 0.8432 |
| Macro-Sensitivity | 0.8432 |
| Macro-Specificity | 0.9748 |
| Macro-Precision | 0.8423 |
| AUC-PR (Macro) | 0.8982 |
| Cohen's Kappa | 0.8408 |
| MCC | 0.8409 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.52M |
| Pure-Inference FPS | 911.8 (1.10 ms/张) |
| End-to-End FPS | 381.7 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8200 | 0.9966 | 0.8119 | 0.8159 | 0.9968 | 0.9957 | 0.9009 | 100 |
| cancer | 0.8969 | 0.9700 | 0.9010 | 0.8989 | 0.9687 | 0.9876 | 0.9632 | 1329 |
| empty | 0.9901 | 0.9975 | 0.9938 | 0.9920 | 0.9961 | 0.9997 | 0.9994 | 1621 |
| normal | 0.8807 | 0.9844 | 0.9244 | 0.9020 | 0.9745 | 0.9897 | 0.9627 | 1014 |
| other | 0.7827 | 0.9482 | 0.7335 | 0.7573 | 0.9599 | 0.9462 | 0.8605 | 879 |
| stroma | 0.6891 | 0.9523 | 0.6891 | 0.6891 | 0.9523 | 0.9529 | 0.7028 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=82 FP=19 TN=5583 FN=18
- **cancer**: TP=1192 FP=131 TN=4242 FN=137
- **empty**: TP=1605 FP=10 TN=4071 FN=16
- **normal**: TP=893 FP=73 TN=4615 FN=121
- **other**: TP=688 FP=250 TN=4573 FN=191
- **stroma**: TP=523 FP=236 TN=4707 FN=236

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 82 | 3 | 1 | 2 | 6 | 6 |
| cancer | 9 | 1192 | 0 | 11 | 44 | 73 |
| empty | 0 | 1 | 1605 | 0 | 13 | 2 |
| normal | 2 | 31 | 0 | 893 | 37 | 51 |
| other | 1 | 56 | 9 | 21 | 688 | 104 |
| stroma | 7 | 40 | 0 | 39 | 150 | 523 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8119    0.8200    0.8159       100
      cancer     0.9010    0.8969    0.8989      1329
       empty     0.9938    0.9901    0.9920      1621
      normal     0.9244    0.8807    0.9020      1014
       other     0.7335    0.7827    0.7573       879
      stroma     0.6891    0.6891    0.6891       759

    accuracy                         0.8739      5702
   macro avg     0.8423    0.8432    0.8425      5702
weighted avg     0.8759    0.8739    0.8747      5702

```