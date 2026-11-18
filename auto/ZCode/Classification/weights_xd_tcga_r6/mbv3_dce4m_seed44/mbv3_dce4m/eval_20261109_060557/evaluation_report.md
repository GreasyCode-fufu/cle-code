# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_dce4m_seed44/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8583 |
| Accuracy | 0.8879 |
| Macro-AUROC (OvR) | 0.9814 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8900 |
| Balanced Acc | 0.8657 |
| Macro-Sensitivity | 0.8657 |
| Macro-Specificity | 0.9779 |
| Macro-Precision | 0.8555 |
| AUC-PR (Macro) | 0.9066 |
| Cohen's Kappa | 0.8588 |
| MCC | 0.8598 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 8786.3 (0.11 ms/张) |
| End-to-End FPS | 557.6 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8300 | 0.9957 | 0.7757 | 0.8019 | 0.9970 | 0.9931 | 0.8791 | 100 |
| cancer | 0.8457 | 0.9776 | 0.9198 | 0.8812 | 0.9542 | 0.9812 | 0.9529 | 1329 |
| empty | 0.9951 | 0.9975 | 0.9938 | 0.9945 | 0.9980 | 0.9998 | 0.9995 | 1621 |
| normal | 0.9280 | 0.9917 | 0.9602 | 0.9438 | 0.9845 | 0.9951 | 0.9819 | 1014 |
| other | 0.7600 | 0.9693 | 0.8186 | 0.7882 | 0.9568 | 0.9604 | 0.8709 | 879 |
| stroma | 0.8353 | 0.9353 | 0.6646 | 0.7402 | 0.9737 | 0.9586 | 0.7552 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=83 FP=24 TN=5578 FN=17
- **cancer**: TP=1124 FP=98 TN=4275 FN=205
- **empty**: TP=1613 FP=10 TN=4071 FN=8
- **normal**: TP=941 FP=39 TN=4649 FN=73
- **other**: TP=668 FP=148 TN=4675 FN=211
- **stroma**: TP=634 FP=320 TN=4623 FN=125

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 83 | 2 | 1 | 6 | 3 | 5 |
| cancer | 16 | 1124 | 0 | 16 | 36 | 137 |
| empty | 0 | 1 | 1613 | 0 | 6 | 1 |
| normal | 3 | 14 | 0 | 941 | 23 | 33 |
| other | 2 | 52 | 9 | 4 | 668 | 144 |
| stroma | 3 | 29 | 0 | 13 | 80 | 634 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7757    0.8300    0.8019       100
      cancer     0.9198    0.8457    0.8812      1329
       empty     0.9938    0.9951    0.9945      1621
      normal     0.9602    0.9280    0.9438      1014
       other     0.8186    0.7600    0.7882       879
      stroma     0.6646    0.8353    0.7402       759

    accuracy                         0.8879      5702
   macro avg     0.8555    0.8657    0.8583      5702
weighted avg     0.8959    0.8879    0.8900      5702

```