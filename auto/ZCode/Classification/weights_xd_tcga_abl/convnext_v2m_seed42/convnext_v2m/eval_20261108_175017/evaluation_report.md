# convnext_v2m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_tcga_abl/convnext_v2m_seed42/convnext_v2m/model_best.pth`
- 测试集: `./datasets/TCGA/val`  (共 10634 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8067 |
| Accuracy | 0.8461 |
| Macro-AUROC (OvR) | 0.9633 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8432 |
| Balanced Acc | 0.8127 |
| Macro-Sensitivity | 0.8127 |
| Macro-Specificity | 0.9681 |
| Macro-Precision | 0.8068 |
| AUC-PR (Macro) | 0.8753 |
| Cohen's Kappa | 0.8020 |
| MCC | 0.8030 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 607.2 (1.65 ms/张) |
| End-to-End FPS | 315.9 |
| 测试样本数 | 10634 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| blood | 0.8392 | 0.9941 | 0.7293 | 0.7804 | 0.9969 | 0.9960 | 0.8846 | 199 |
| cancer | 0.8968 | 0.9536 | 0.8647 | 0.8805 | 0.9654 | 0.9818 | 0.9533 | 2645 |
| empty | 0.9969 | 0.9607 | 0.9180 | 0.9558 | 0.9986 | 0.9822 | 0.9518 | 3256 |
| normal | 0.7285 | 0.9847 | 0.9029 | 0.8064 | 0.9490 | 0.9668 | 0.8942 | 1735 |
| other | 0.6575 | 0.9520 | 0.7251 | 0.6897 | 0.9352 | 0.8884 | 0.7633 | 1717 |
| stroma | 0.7569 | 0.9634 | 0.7006 | 0.7277 | 0.9722 | 0.9647 | 0.8046 | 1082 |

### 每类 TP/FP/TN/FN

- **blood**: TP=167 FP=62 TN=10373 FN=32
- **cancer**: TP=2372 FP=371 TN=7618 FN=273
- **empty**: TP=3246 FP=290 TN=7088 FN=10
- **normal**: TP=1264 FP=136 TN=8763 FN=471
- **other**: TP=1129 FP=428 TN=8489 FN=588
- **stroma**: TP=819 FP=350 TN=9202 FN=263

## 混淆矩阵 (行=真实, 列=预测)

| | blood | cancer | empty | normal | other | stroma |
|---|---|---|---|---|---|---|
| blood | 167 | 5 | 2 | 4 | 20 | 1 |
| cancer | 9 | 2372 | 1 | 68 | 152 | 43 |
| empty | 0 | 0 | 3246 | 0 | 10 | 0 |
| normal | 2 | 152 | 0 | 1264 | 124 | 193 |
| other | 46 | 128 | 287 | 14 | 1129 | 113 |
| stroma | 5 | 86 | 0 | 50 | 122 | 819 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       blood     0.7293    0.8392    0.7804       199
      cancer     0.8647    0.8968    0.8805      2645
       empty     0.9180    0.9969    0.9558      3256
      normal     0.9029    0.7285    0.8064      1735
       other     0.7251    0.6575    0.6897      1717
      stroma     0.7006    0.7569    0.7277      1082

    accuracy                         0.8461     10634
   macro avg     0.8068    0.8127    0.8067     10634
weighted avg     0.8455    0.8461    0.8432     10634

```