# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce16_seed43/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8352 |
| Accuracy | 0.8734 |
| Macro-AUROC (OvR) | 0.9759 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8744 |
| Balanced Acc | 0.8317 |
| Macro-Sensitivity | 0.8317 |
| Macro-Specificity | 0.9746 |
| Macro-Precision | 0.8443 |
| AUC-PR (Macro) | 0.8739 |
| Cohen's Kappa | 0.8400 |
| MCC | 0.8410 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9153.9 (0.11 ms/张) |
| End-to-End FPS | 552.9 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7300 | 0.9963 | 0.7766 | 0.7526 | 0.9952 | 0.9877 | 0.7902 | 100 |
| cancer | 0.8841 | 0.9545 | 0.8552 | 0.8694 | 0.9644 | 0.9780 | 0.9455 | 1329 |
| empty | 0.9957 | 0.9963 | 0.9908 | 0.9932 | 0.9983 | 0.9999 | 0.9998 | 1621 |
| normal | 0.8935 | 0.9900 | 0.9507 | 0.9212 | 0.9773 | 0.9933 | 0.9787 | 1014 |
| other | 0.6951 | 0.9770 | 0.8463 | 0.7633 | 0.9462 | 0.9450 | 0.8542 | 879 |
| stroma | 0.7918 | 0.9334 | 0.6462 | 0.7117 | 0.9669 | 0.9513 | 0.6751 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=73 FP=21 TN=5581 FN=27
- **cancer**: TP=1175 FP=199 TN=4174 FN=154
- **empty**: TP=1614 FP=15 TN=4066 FN=7
- **normal**: TP=906 FP=47 TN=4641 FN=108
- **other**: TP=611 FP=111 TN=4712 FN=268
- **stroma**: TP=601 FP=329 TN=4614 FN=158

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 73 | 4 | 2 | 6 | 7 | 8 |
| cancer | 18 | 1175 | 1 | 17 | 30 | 88 |
| empty | 0 | 0 | 1614 | 1 | 5 | 1 |
| normal | 1 | 27 | 0 | 906 | 30 | 50 |
| other | 0 | 66 | 11 | 9 | 611 | 182 |
| stroma | 2 | 102 | 1 | 14 | 39 | 601 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7766    0.7300    0.7526       100
      cancer     0.8552    0.8841    0.8694      1329
       empty     0.9908    0.9957    0.9932      1621
      normal     0.9507    0.8935    0.9212      1014
       other     0.8463    0.6951    0.7633       879
      stroma     0.6462    0.7918    0.7117       759

    accuracy                         0.8734      5702
   macro avg     0.8443    0.8317    0.8352      5702
weighted avg     0.8801    0.8734    0.8744      5702

```