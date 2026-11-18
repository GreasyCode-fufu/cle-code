# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_fz16_dce16_seed44/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8406 |
| Accuracy | 0.8741 |
| Macro-AUROC (OvR) | 0.9748 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8764 |
| Balanced Acc | 0.8359 |
| Macro-Sensitivity | 0.8359 |
| Macro-Specificity | 0.9749 |
| Macro-Precision | 0.8493 |
| AUC-PR (Macro) | 0.8795 |
| Cohen's Kappa | 0.8411 |
| MCC | 0.8418 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.69M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 9146.8 (0.11 ms/张) |
| End-to-End FPS | 558.4 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7400 | 0.9971 | 0.8222 | 0.7789 | 0.9954 | 0.9919 | 0.8272 | 100 |
| cancer | 0.8698 | 0.9664 | 0.8872 | 0.8784 | 0.9607 | 0.9765 | 0.9471 | 1329 |
| empty | 0.9944 | 0.9973 | 0.9932 | 0.9938 | 0.9978 | 0.9999 | 0.9998 | 1621 |
| normal | 0.8688 | 0.9915 | 0.9566 | 0.9106 | 0.9722 | 0.9888 | 0.9696 | 1014 |
| other | 0.7543 | 0.9639 | 0.7921 | 0.7727 | 0.9556 | 0.9441 | 0.8575 | 879 |
| stroma | 0.7879 | 0.9332 | 0.6444 | 0.7090 | 0.9663 | 0.9474 | 0.6759 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=74 FP=16 TN=5586 FN=26
- **cancer**: TP=1156 FP=147 TN=4226 FN=173
- **empty**: TP=1612 FP=11 TN=4070 FN=9
- **normal**: TP=881 FP=40 TN=4648 FN=133
- **other**: TP=663 FP=174 TN=4649 FN=216
- **stroma**: TP=598 FP=330 TN=4613 FN=161

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 74 | 5 | 1 | 5 | 9 | 6 |
| cancer | 9 | 1156 | 1 | 14 | 44 | 105 |
| empty | 0 | 0 | 1612 | 0 | 7 | 2 |
| normal | 4 | 37 | 0 | 881 | 34 | 58 |
| other | 2 | 40 | 8 | 7 | 663 | 159 |
| stroma | 1 | 65 | 1 | 14 | 80 | 598 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8222    0.7400    0.7789       100
      cancer     0.8872    0.8698    0.8784      1329
       empty     0.9932    0.9944    0.9938      1621
      normal     0.9566    0.8688    0.9106      1014
       other     0.7921    0.7543    0.7727       879
      stroma     0.6444    0.7879    0.7090       759

    accuracy                         0.8741      5702
   macro avg     0.8493    0.8359    0.8406      5702
weighted avg     0.8816    0.8741    0.8764      5702

```