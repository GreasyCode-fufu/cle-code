# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_tcga_ep/convnext_egb_scg_seed42/convnext_egb_scg/model_best.pth`
- 测试集: `./../../datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8100 |
| Accuracy | 0.8488 |
| Macro-AUROC (OvR) | 0.9654 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8428 |
| Balanced Acc | 0.8087 |
| Macro-Sensitivity | 0.8087 |
| Macro-Specificity | 0.9683 |
| Macro-Precision | 0.8175 |
| AUC-PR (Macro) | 0.8803 |
| Cohen's Kappa | 0.8049 |
| MCC | 0.8066 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.29M |
| Pure-Inference FPS | 344.4 (2.90 ms/张) |
| End-to-End FPS | 227.5 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8291 | 0.9959 | 0.7933 | 0.8108 | 0.9967 | 0.9978 | 0.9212 | 199 |
| cancer | 0.9270 | 0.9423 | 0.8417 | 0.8823 | 0.9750 | 0.9854 | 0.9599 | 2645 |
| empty | 0.9966 | 0.9607 | 0.9180 | 0.9557 | 0.9985 | 0.9825 | 0.9526 | 3256 |
| normal | 0.8202 | 0.9783 | 0.8806 | 0.8493 | 0.9654 | 0.9833 | 0.9347 | 1735 |
| other | 0.5620 | 0.9639 | 0.7498 | 0.6425 | 0.9195 | 0.8771 | 0.7289 | 1717 |
| stroma | 0.7172 | 0.9687 | 0.7219 | 0.7195 | 0.9680 | 0.9661 | 0.7847 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=165 FP=43 TN=10392 FN=34
- **cancer**: TP=2452 FP=461 TN=7528 FN=193
- **empty**: TP=3245 FP=290 TN=7088 FN=11
- **normal**: TP=1423 FP=193 TN=8706 FN=312
- **other**: TP=965 FP=322 TN=8595 FN=752
- **stroma**: TP=776 FP=299 TN=9253 FN=306

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 165 | 5 | 2 | 4 | 22 | 1 |
| cancer | 2 | 2452 | 0 | 63 | 78 | 50 |
| empty | 0 | 0 | 3245 | 0 | 11 | 0 |
| normal | 2 | 133 | 0 | 1423 | 81 | 96 |
| other | 35 | 229 | 288 | 48 | 965 | 152 |
| stroma | 4 | 94 | 0 | 78 | 130 | 776 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7933    0.8291    0.8108       199
      cancer     0.8417    0.9270    0.8823      2645
       empty     0.9180    0.9966    0.9557      3256
      normal     0.8806    0.8202    0.8493      1735
       other     0.7498    0.5620    0.6425      1717
      stroma     0.7219    0.7172    0.7195      1082

    accuracy                         0.8488     10634
   macro avg     0.8175    0.8087    0.8100     10634
weighted avg     0.8435    0.8488    0.8428     10634

```