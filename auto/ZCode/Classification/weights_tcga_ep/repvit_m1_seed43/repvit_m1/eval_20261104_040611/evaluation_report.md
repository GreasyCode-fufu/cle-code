# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/repvit_m1_seed43/repvit_m1/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7762 |
| Accuracy | 0.8195 |
| Macro-AUROC (OvR) | 0.9404 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8177 |
| Balanced Acc | 0.7729 |
| Macro-Sensitivity | 0.7729 |
| Macro-Specificity | 0.9629 |
| Macro-Precision | 0.7871 |
| AUC-PR (Macro) | 0.8223 |
| Cohen's Kappa | 0.7677 |
| MCC | 0.7693 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3307.2 (0.30 ms/张) |
| End-to-End FPS | 542.5 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.7940 | 0.9957 | 0.7783 | 0.7861 | 0.9961 | 0.9898 | 0.8700 | 199 |
| cancer | 0.8934 | 0.9666 | 0.8985 | 0.8959 | 0.9648 | 0.9841 | 0.9608 | 2645 |
| empty | 0.9975 | 0.9561 | 0.9093 | 0.9514 | 0.9989 | 0.9752 | 0.9073 | 3256 |
| normal | 0.6450 | 0.9846 | 0.8909 | 0.7482 | 0.9343 | 0.8886 | 0.8142 | 1735 |
| other | 0.6488 | 0.9139 | 0.5919 | 0.6191 | 0.9311 | 0.8613 | 0.6898 | 1717 |
| stroma | 0.6590 | 0.9604 | 0.6535 | 0.6562 | 0.9613 | 0.9438 | 0.6918 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=158 FP=45 TN=10390 FN=41
- **cancer**: TP=2363 FP=267 TN=7722 FN=282
- **empty**: TP=3248 FP=324 TN=7054 FN=8
- **normal**: TP=1119 FP=137 TN=8762 FN=616
- **other**: TP=1114 FP=768 TN=8149 FN=603
- **stroma**: TP=713 FP=378 TN=9174 FN=369

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 158 | 5 | 6 | 3 | 26 | 1 |
| cancer | 4 | 2363 | 3 | 35 | 183 | 57 |
| empty | 0 | 1 | 3248 | 0 | 7 | 0 |
| normal | 7 | 70 | 2 | 1119 | 336 | 201 |
| other | 33 | 101 | 312 | 38 | 1114 | 119 |
| stroma | 1 | 90 | 1 | 61 | 216 | 713 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7783    0.7940    0.7861       199
      cancer     0.8985    0.8934    0.8959      2645
       empty     0.9093    0.9975    0.9514      3256
      normal     0.8909    0.6450    0.7482      1735
       other     0.5919    0.6488    0.6191      1717
      stroma     0.6535    0.6590    0.6562      1082

    accuracy                         0.8195     10634
   macro avg     0.7871    0.7729    0.7762     10634
weighted avg     0.8239    0.8195    0.8177     10634

```