# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/repvit_m1_seed44/repvit_m1/model_best.pth`
- 测试集: `./../../datasets/TCGA/test`  (共 5702 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7913 |
| Accuracy | 0.8259 |
| Macro-AUROC (OvR) | 0.9677 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8287 |
| Balanced Acc | 0.8232 |
| Macro-Sensitivity | 0.8232 |
| Macro-Specificity | 0.9658 |
| Macro-Precision | 0.7831 |
| AUC-PR (Macro) | 0.8582 |
| Cohen's Kappa | 0.7819 |
| MCC | 0.7859 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 4827.3 (0.21 ms/张) |
| End-to-End FPS | 592.9 |
| 测试样本数 | 5702 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8800 | 0.9905 | 0.6241 | 0.7303 | 0.9978 | 0.9940 | 0.8288 | 100 |
| cancer | 0.7163 | 0.9792 | 0.9128 | 0.8027 | 0.9191 | 0.9592 | 0.9036 | 1329 |
| empty | 0.9951 | 0.9971 | 0.9926 | 0.9938 | 0.9980 | 0.9996 | 0.9993 | 1621 |
| normal | 0.7535 | 0.9735 | 0.8604 | 0.8034 | 0.9481 | 0.9767 | 0.8936 | 1014 |
| other | 0.6815 | 0.9507 | 0.7157 | 0.6981 | 0.9424 | 0.9170 | 0.7958 | 879 |
| stroma | 0.9130 | 0.9039 | 0.5933 | 0.7193 | 0.9854 | 0.9598 | 0.7282 | 759 |

### 每类 TP/FP/TN/FN

- **blood**: TP=88 FP=53 TN=5549 FN=12
- **cancer**: TP=952 FP=91 TN=4282 FN=377
- **empty**: TP=1613 FP=12 TN=4069 FN=8
- **normal**: TP=764 FP=124 TN=4564 FN=250
- **other**: TP=599 FP=238 TN=4585 FN=280
- **stroma**: TP=693 FP=475 TN=4468 FN=66

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 88 | 1 | 1 | 1 | 5 | 4 |
| cancer | 22 | 952 | 1 | 105 | 28 | 221 |
| empty | 1 | 2 | 1613 | 0 | 4 | 1 |
| normal | 10 | 24 | 0 | 764 | 175 | 41 |
| other | 10 | 48 | 9 | 5 | 599 | 208 |
| stroma | 10 | 16 | 1 | 13 | 26 | 693 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.6241    0.8800    0.7303       100
      cancer     0.9128    0.7163    0.8027      1329
       empty     0.9926    0.9951    0.9938      1621
      normal     0.8604    0.7535    0.8034      1014
       other     0.7157    0.6815    0.6981       879
      stroma     0.5933    0.9130    0.7193       759

    accuracy                         0.8259      5702
   macro avg     0.7831    0.8232    0.7913      5702
weighted avg     0.8482    0.8259    0.8287      5702

```