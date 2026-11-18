# convnext_tiny_cle_oca_strip 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_tiny_cle_oca_strip_seed44/convnext_tiny_cle_oca_strip/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8610 |
| Accuracy | 0.8971 |
| Macro-AUROC (OvR) | 0.9816 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8971 |
| Balanced Acc | 0.8541 |
| Macro-Sensitivity | 0.8541 |
| Macro-Specificity | 0.9793 |
| Macro-Precision | 0.8693 |
| AUC-PR (Macro) | 0.9114 |
| Cohen's Kappa | 0.8699 |
| MCC | 0.8699 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.47G |
| FLOPs | 8.94G |
| Params | 27.84M |
| Pure-Inference FPS | 1554.8 (0.64 ms/张) |
| End-to-End FPS | 473.0 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7300 | 0.9973 | 0.8295 | 0.7766 | 0.9952 | 0.9914 | 0.8691 | 100 |
| cancer | 0.8977 | 0.9668 | 0.8916 | 0.8946 | 0.9688 | 0.9802 | 0.9521 | 1329 |
| empty | 0.9957 | 0.9971 | 0.9926 | 0.9941 | 0.9983 | 0.9998 | 0.9994 | 1621 |
| normal | 0.9507 | 0.9895 | 0.9516 | 0.9512 | 0.9893 | 0.9960 | 0.9846 | 1014 |
| other | 0.7861 | 0.9699 | 0.8266 | 0.8058 | 0.9614 | 0.9592 | 0.8893 | 879 |
| stroma | 0.7642 | 0.9553 | 0.7241 | 0.7436 | 0.9635 | 0.9633 | 0.7739 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=73 FP=15 TN=5587 FN=27
- **cancer**: TP=1193 FP=145 TN=4228 FN=136
- **empty**: TP=1614 FP=12 TN=4069 FN=7
- **normal**: TP=964 FP=49 TN=4639 FN=50
- **other**: TP=691 FP=145 TN=4678 FN=188
- **stroma**: TP=580 FP=221 TN=4722 FN=179

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 73 | 3 | 2 | 4 | 13 | 5 |
| cancer | 11 | 1193 | 2 | 14 | 24 | 85 |
| empty | 0 | 1 | 1614 | 0 | 6 | 0 |
| normal | 1 | 24 | 0 | 964 | 4 | 21 |
| other | 1 | 63 | 7 | 7 | 691 | 110 |
| stroma | 2 | 54 | 1 | 24 | 98 | 580 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8295    0.7300    0.7766       100
      cancer     0.8916    0.8977    0.8946      1329
       empty     0.9926    0.9957    0.9941      1621
      normal     0.9516    0.9507    0.9512      1014
       other     0.8266    0.7861    0.8058       879
      stroma     0.7241    0.7642    0.7436       759

    accuracy                         0.8971      5702
   macro avg     0.8693    0.8541    0.8610      5702
weighted avg     0.8976    0.8971    0.8971      5702

```