# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_r6_fz16_seed44/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8296 |
| Accuracy | 0.8641 |
| Macro-AUROC (OvR) | 0.9726 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8653 |
| Balanced Acc | 0.8310 |
| Macro-Sensitivity | 0.8310 |
| Macro-Specificity | 0.9727 |
| Macro-Precision | 0.8299 |
| AUC-PR (Macro) | 0.8736 |
| Cohen's Kappa | 0.8282 |
| MCC | 0.8286 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.92M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 9179.5 (0.11 ms/张) |
| End-to-End FPS | 629.4 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8000 | 0.9955 | 0.7619 | 0.7805 | 0.9964 | 0.9912 | 0.8230 | 100 |
| cancer | 0.8819 | 0.9488 | 0.8395 | 0.8602 | 0.9635 | 0.9742 | 0.9391 | 1329 |
| empty | 0.9951 | 0.9973 | 0.9932 | 0.9941 | 0.9980 | 0.9997 | 0.9995 | 1621 |
| normal | 0.8550 | 0.9919 | 0.9580 | 0.9036 | 0.9694 | 0.9893 | 0.9646 | 1014 |
| other | 0.7634 | 0.9623 | 0.7866 | 0.7748 | 0.9571 | 0.9425 | 0.8602 | 879 |
| stroma | 0.6904 | 0.9403 | 0.6398 | 0.6641 | 0.9519 | 0.9390 | 0.6552 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=80 FP=25 TN=5577 FN=20
- **cancer**: TP=1172 FP=224 TN=4149 FN=157
- **empty**: TP=1613 FP=11 TN=4070 FN=8
- **normal**: TP=867 FP=38 TN=4650 FN=147
- **other**: TP=671 FP=182 TN=4641 FN=208
- **stroma**: TP=524 FP=295 TN=4648 FN=235

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 80 | 4 | 0 | 4 | 7 | 5 |
| cancer | 13 | 1172 | 1 | 9 | 46 | 88 |
| empty | 0 | 1 | 1613 | 0 | 5 | 2 |
| normal | 3 | 58 | 0 | 867 | 25 | 61 |
| other | 5 | 46 | 9 | 9 | 671 | 139 |
| stroma | 4 | 115 | 1 | 16 | 99 | 524 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7619    0.8000    0.7805       100
      cancer     0.8395    0.8819    0.8602      1329
       empty     0.9932    0.9951    0.9941      1621
      normal     0.9580    0.8550    0.9036      1014
       other     0.7866    0.7634    0.7748       879
      stroma     0.6398    0.6904    0.6641       759

    accuracy                         0.8641      5702
   macro avg     0.8299    0.8310    0.8296      5702
weighted avg     0.8682    0.8641    0.8653      5702

```