# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_r6/mbv3_r6_fz16_seed43/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7860 |
| Accuracy | 0.8291 |
| Macro-AUROC (OvR) | 0.9474 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8230 |
| Balanced Acc | 0.7864 |
| Macro-Sensitivity | 0.7864 |
| Macro-Specificity | 0.9643 |
| Macro-Precision | 0.7922 |
| AUC-PR (Macro) | 0.8327 |
| Cohen's Kappa | 0.7793 |
| MCC | 0.7809 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.92M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 7542.5 (0.13 ms/张) |
| End-to-End FPS | 33.2 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8593 | 0.9950 | 0.7668 | 0.8104 | 0.9973 | 0.9923 | 0.8797 | 199 |
| cancer | 0.9255 | 0.9298 | 0.8136 | 0.8659 | 0.9742 | 0.9749 | 0.9121 | 2645 |
| empty | 0.9960 | 0.9631 | 0.9226 | 0.9579 | 0.9982 | 0.9919 | 0.9757 | 3256 |
| normal | 0.7452 | 0.9767 | 0.8620 | 0.7994 | 0.9516 | 0.9236 | 0.8459 | 1735 |
| other | 0.5859 | 0.9528 | 0.7050 | 0.6399 | 0.9228 | 0.8799 | 0.7082 | 1717 |
| stroma | 0.6063 | 0.9682 | 0.6833 | 0.6425 | 0.9560 | 0.9216 | 0.6748 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=171 FP=52 TN=10383 FN=28
- **cancer**: TP=2448 FP=561 TN=7428 FN=197
- **empty**: TP=3243 FP=272 TN=7106 FN=13
- **normal**: TP=1293 FP=207 TN=8692 FN=442
- **other**: TP=1006 FP=421 TN=8496 FN=711
- **stroma**: TP=656 FP=304 TN=9248 FN=426

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 171 | 6 | 2 | 7 | 12 | 1 |
| cancer | 6 | 2448 | 0 | 47 | 88 | 56 |
| empty | 0 | 1 | 3243 | 0 | 12 | 0 |
| normal | 1 | 200 | 1 | 1293 | 129 | 111 |
| other | 41 | 229 | 268 | 37 | 1006 | 136 |
| stroma | 4 | 125 | 1 | 116 | 180 | 656 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7668    0.8593    0.8104       199
      cancer     0.8136    0.9255    0.8659      2645
       empty     0.9226    0.9960    0.9579      3256
      normal     0.8620    0.7452    0.7994      1735
       other     0.7050    0.5859    0.6399      1717
      stroma     0.6833    0.6063    0.6425      1082

    accuracy                         0.8291     10634
   macro avg     0.7922    0.7864    0.7860     10634
weighted avg     0.8232    0.8291    0.8230     10634

```