# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r5/convnext_v6f_seed44/convnext_v6f/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8456 |
| Accuracy | 0.8741 |
| Macro-AUROC (OvR) | 0.9782 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8766 |
| Balanced Acc | 0.8741 |
| Macro-Sensitivity | 0.8741 |
| Macro-Specificity | 0.9756 |
| Macro-Precision | 0.8395 |
| AUC-PR (Macro) | 0.9113 |
| Cohen's Kappa | 0.8423 |
| MCC | 0.8468 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.26M |
| Pure-Inference FPS | 633.7 (1.58 ms/张) |
| End-to-End FPS | 322.4 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.9100 | 0.9929 | 0.6947 | 0.7879 | 0.9984 | 0.9973 | 0.9078 | 100 |
| cancer | 0.7223 | 0.9938 | 0.9726 | 0.8290 | 0.9217 | 0.9691 | 0.9342 | 1329 |
| empty | 0.9957 | 0.9985 | 0.9963 | 0.9960 | 0.9983 | 0.9999 | 0.9997 | 1621 |
| normal | 0.9645 | 0.9885 | 0.9477 | 0.9560 | 0.9923 | 0.9969 | 0.9895 | 1014 |
| other | 0.7258 | 0.9695 | 0.8127 | 0.7668 | 0.9510 | 0.9399 | 0.8436 | 879 |
| stroma | 0.9262 | 0.9102 | 0.6129 | 0.7377 | 0.9877 | 0.9664 | 0.7932 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=91 FP=40 TN=5562 FN=9
- **cancer**: TP=960 FP=27 TN=4346 FN=369
- **empty**: TP=1614 FP=6 TN=4075 FN=7
- **normal**: TP=978 FP=54 TN=4634 FN=36
- **other**: TP=638 FP=147 TN=4676 FN=241
- **stroma**: TP=703 FP=444 TN=4499 FN=56

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 91 | 1 | 0 | 2 | 5 | 1 |
| cancer | 29 | 960 | 0 | 26 | 102 | 212 |
| empty | 1 | 0 | 1614 | 0 | 4 | 2 |
| normal | 4 | 8 | 0 | 978 | 1 | 23 |
| other | 3 | 14 | 6 | 12 | 638 | 206 |
| stroma | 3 | 4 | 0 | 14 | 35 | 703 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6947    0.9100    0.7879       100
      cancer     0.9726    0.7223    0.8290      1329
       empty     0.9963    0.9957    0.9960      1621
      normal     0.9477    0.9645    0.9560      1014
       other     0.8127    0.7258    0.7668       879
      stroma     0.6129    0.9262    0.7377       759

    accuracy                         0.8741      5702
   macro avg     0.8395    0.8741    0.8456      5702
weighted avg     0.8975    0.8741    0.8766      5702

```