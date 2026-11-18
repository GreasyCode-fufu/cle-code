# convnext_egb_scg_hsc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_egb_scg_hsc_seed42/convnext_egb_scg_hsc/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8074 |
| Accuracy | 0.8397 |
| Macro-AUROC (OvR) | 0.9675 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8346 |
| Balanced Acc | 0.8001 |
| Macro-Sensitivity | 0.8001 |
| Macro-Specificity | 0.9664 |
| Macro-Precision | 0.8186 |
| AUC-PR (Macro) | 0.8781 |
| Cohen's Kappa | 0.7930 |
| MCC | 0.7941 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.58M |
| Pure-Inference FPS | 344.1 (2.91 ms/张) |
| End-to-End FPS | 227.5 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8492 | 0.9972 | 0.8535 | 0.8514 | 0.9971 | 0.9981 | 0.9379 | 199 |
| cancer | 0.9316 | 0.9437 | 0.8456 | 0.8865 | 0.9766 | 0.9831 | 0.9558 | 2645 |
| empty | 0.9926 | 0.9638 | 0.9237 | 0.9569 | 0.9966 | 0.9861 | 0.9677 | 3256 |
| normal | 0.7481 | 0.9697 | 0.8278 | 0.7860 | 0.9518 | 0.9524 | 0.8683 | 1735 |
| other | 0.6022 | 0.9461 | 0.6825 | 0.6399 | 0.9251 | 0.9217 | 0.7235 | 1717 |
| stroma | 0.6765 | 0.9782 | 0.7787 | 0.7240 | 0.9639 | 0.9633 | 0.8152 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=169 FP=29 TN=10406 FN=30
- **cancer**: TP=2464 FP=450 TN=7539 FN=181
- **empty**: TP=3232 FP=267 TN=7111 FN=24
- **normal**: TP=1298 FP=270 TN=8629 FN=437
- **other**: TP=1034 FP=481 TN=8436 FN=683
- **stroma**: TP=732 FP=208 TN=9344 FN=350

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 169 | 5 | 2 | 8 | 14 | 1 |
| cancer | 2 | 2464 | 0 | 114 | 27 | 38 |
| empty | 0 | 0 | 3232 | 0 | 24 | 0 |
| normal | 0 | 142 | 0 | 1298 | 235 | 60 |
| other | 23 | 231 | 265 | 55 | 1034 | 109 |
| stroma | 4 | 72 | 0 | 93 | 181 | 732 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.8535    0.8492    0.8514       199
      cancer     0.8456    0.9316    0.8865      2645
       empty     0.9237    0.9926    0.9569      3256
      normal     0.8278    0.7481    0.7860      1735
       other     0.6825    0.6022    0.6399      1717
      stroma     0.7787    0.6765    0.7240      1082

    accuracy                         0.8397     10634
   macro avg     0.8186    0.8001    0.8074     10634
weighted avg     0.8336    0.8397    0.8346     10634

```