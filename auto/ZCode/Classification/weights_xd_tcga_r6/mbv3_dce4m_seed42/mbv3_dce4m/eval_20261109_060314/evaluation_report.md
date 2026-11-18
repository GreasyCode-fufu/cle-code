# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_dce4m_seed42/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8503 |
| Accuracy | 0.8799 |
| Macro-AUROC (OvR) | 0.9802 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8819 |
| Balanced Acc | 0.8563 |
| Macro-Sensitivity | 0.8563 |
| Macro-Specificity | 0.9761 |
| Macro-Precision | 0.8466 |
| AUC-PR (Macro) | 0.9014 |
| Cohen's Kappa | 0.8486 |
| MCC | 0.8491 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 8762.1 (0.11 ms/张) |
| End-to-End FPS | 561.5 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8200 | 0.9957 | 0.7736 | 0.7961 | 0.9968 | 0.9935 | 0.8668 | 100 |
| cancer | 0.8322 | 0.9726 | 0.9021 | 0.8658 | 0.9502 | 0.9805 | 0.9480 | 1329 |
| empty | 0.9957 | 0.9980 | 0.9951 | 0.9954 | 0.9983 | 0.9999 | 0.9997 | 1621 |
| normal | 0.9014 | 0.9932 | 0.9662 | 0.9327 | 0.9790 | 0.9948 | 0.9825 | 1014 |
| other | 0.7952 | 0.9467 | 0.7312 | 0.7619 | 0.9621 | 0.9507 | 0.8530 | 879 |
| stroma | 0.7931 | 0.9506 | 0.7116 | 0.7502 | 0.9677 | 0.9619 | 0.7585 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=82 FP=24 TN=5578 FN=18
- **cancer**: TP=1106 FP=120 TN=4253 FN=223
- **empty**: TP=1614 FP=8 TN=4073 FN=7
- **normal**: TP=914 FP=32 TN=4656 FN=100
- **other**: TP=699 FP=257 TN=4566 FN=180
- **stroma**: TP=602 FP=244 TN=4699 FN=157

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 82 | 1 | 2 | 4 | 7 | 4 |
| cancer | 16 | 1106 | 0 | 15 | 117 | 75 |
| empty | 0 | 0 | 1614 | 0 | 5 | 2 |
| normal | 2 | 48 | 0 | 914 | 12 | 38 |
| other | 4 | 41 | 6 | 4 | 699 | 125 |
| stroma | 2 | 30 | 0 | 9 | 116 | 602 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7736    0.8200    0.7961       100
      cancer     0.9021    0.8322    0.8658      1329
       empty     0.9951    0.9957    0.9954      1621
      normal     0.9662    0.9014    0.9327      1014
       other     0.7312    0.7952    0.7619       879
      stroma     0.7116    0.7931    0.7502       759

    accuracy                         0.8799      5702
   macro avg     0.8466    0.8563    0.8503      5702
weighted avg     0.8860    0.8799    0.8819      5702

```