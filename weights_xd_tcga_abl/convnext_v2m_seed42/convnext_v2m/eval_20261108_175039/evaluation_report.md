# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_abl/convnext_v2m_seed42/convnext_v2m/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8622 |
| Accuracy | 0.8907 |
| Macro-AUROC (OvR) | 0.9815 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8919 |
| Balanced Acc | 0.8797 |
| Macro-Sensitivity | 0.8797 |
| Macro-Specificity | 0.9785 |
| Macro-Precision | 0.8548 |
| AUC-PR (Macro) | 0.9085 |
| Cohen's Kappa | 0.8626 |
| MCC | 0.8646 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 611.0 (1.64 ms/张) |
| End-to-End FPS | 318.5 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8700 | 0.9945 | 0.7373 | 0.7982 | 0.9977 | 0.9959 | 0.8788 | 100 |
| cancer | 0.7886 | 0.9861 | 0.9450 | 0.8597 | 0.9388 | 0.9761 | 0.9429 | 1329 |
| empty | 0.9963 | 0.9956 | 0.9890 | 0.9926 | 0.9985 | 0.9998 | 0.9996 | 1621 |
| normal | 0.9714 | 0.9889 | 0.9499 | 0.9605 | 0.9938 | 0.9967 | 0.9889 | 1014 |
| other | 0.7531 | 0.9724 | 0.8327 | 0.7909 | 0.9558 | 0.9532 | 0.8690 | 879 |
| stroma | 0.8986 | 0.9336 | 0.6752 | 0.7711 | 0.9836 | 0.9671 | 0.7717 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=87 FP=31 TN=5571 FN=13
- **cancer**: TP=1048 FP=61 TN=4312 FN=281
- **empty**: TP=1615 FP=18 TN=4063 FN=6
- **normal**: TP=985 FP=52 TN=4636 FN=29
- **other**: TP=662 FP=133 TN=4690 FN=217
- **stroma**: TP=682 FP=328 TN=4615 FN=77

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 87 | 0 | 4 | 4 | 4 | 1 |
| cancer | 25 | 1048 | 0 | 22 | 78 | 156 |
| empty | 0 | 0 | 1615 | 1 | 4 | 1 |
| normal | 2 | 6 | 0 | 985 | 2 | 19 |
| other | 1 | 42 | 14 | 9 | 662 | 151 |
| stroma | 3 | 13 | 0 | 16 | 45 | 682 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7373    0.8700    0.7982       100
      cancer     0.9450    0.7886    0.8597      1329
       empty     0.9890    0.9963    0.9926      1621
      normal     0.9499    0.9714    0.9605      1014
       other     0.8327    0.7531    0.7909       879
      stroma     0.6752    0.8986    0.7711       759

    accuracy                         0.8907      5702
   macro avg     0.8548    0.8797    0.8622      5702
weighted avg     0.9015    0.8907    0.8919      5702

```