# cle_convnext_sn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/cle_convnext_sn_seed44/cle_convnext_sn/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8469 |
| Accuracy | 0.8876 |
| Macro-AUROC (OvR) | 0.9818 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8874 |
| Balanced Acc | 0.8531 |
| Macro-Sensitivity | 0.8531 |
| Macro-Specificity | 0.9776 |
| Macro-Precision | 0.8418 |
| AUC-PR (Macro) | 0.9096 |
| Cohen's Kappa | 0.8582 |
| MCC | 0.8584 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 601.8 (1.66 ms/张) |
| End-to-End FPS | 316.3 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7700 | 0.9945 | 0.7130 | 0.7404 | 0.9959 | 0.9953 | 0.8644 | 100 |
| cancer | 0.8518 | 0.9739 | 0.9085 | 0.8792 | 0.9558 | 0.9794 | 0.9439 | 1329 |
| empty | 0.9907 | 0.9985 | 0.9963 | 0.9935 | 0.9963 | 0.9999 | 0.9998 | 1621 |
| normal | 0.9803 | 0.9802 | 0.9144 | 0.9462 | 0.9957 | 0.9971 | 0.9880 | 1014 |
| other | 0.7838 | 0.9544 | 0.7580 | 0.7707 | 0.9604 | 0.9556 | 0.8612 | 879 |
| stroma | 0.7418 | 0.9642 | 0.7608 | 0.7512 | 0.9605 | 0.9635 | 0.8005 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=77 FP=31 TN=5571 FN=23
- **cancer**: TP=1132 FP=114 TN=4259 FN=197
- **empty**: TP=1606 FP=6 TN=4075 FN=15
- **normal**: TP=994 FP=93 TN=4595 FN=20
- **other**: TP=689 FP=220 TN=4603 FN=190
- **stroma**: TP=563 FP=177 TN=4766 FN=196

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 77 | 3 | 2 | 7 | 8 | 3 |
| cancer | 26 | 1132 | 0 | 45 | 70 | 56 |
| empty | 0 | 0 | 1606 | 0 | 13 | 2 |
| normal | 2 | 15 | 0 | 994 | 0 | 3 |
| other | 2 | 63 | 4 | 8 | 689 | 113 |
| stroma | 1 | 33 | 0 | 33 | 129 | 563 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7130    0.7700    0.7404       100
      cancer     0.9085    0.8518    0.8792      1329
       empty     0.9963    0.9907    0.9935      1621
      normal     0.9144    0.9803    0.9462      1014
       other     0.7580    0.7838    0.7707       879
      stroma     0.7608    0.7418    0.7512       759

    accuracy                         0.8876      5702
   macro avg     0.8418    0.8531    0.8469      5702
weighted avg     0.8882    0.8876    0.8874      5702

```