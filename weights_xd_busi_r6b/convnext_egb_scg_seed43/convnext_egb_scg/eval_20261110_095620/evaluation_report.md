# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/convnext_egb_scg_seed43/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8524 |
| Accuracy | 0.8879 |
| Macro-AUROC (OvR) | 0.9539 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8864 |
| Balanced Acc | 0.8537 |
| Macro-Sensitivity | 0.8537 |
| Macro-Specificity | 0.9408 |
| Macro-Precision | 0.8550 |
| AUC-PR (Macro) | 0.9141 |
| Cohen's Kappa | 0.8073 |
| MCC | 0.8085 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 356.3 (2.81 ms/张) |
| End-to-End FPS | 123.8 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9692 | 0.9216 | 0.9403 | 0.9545 | 0.9592 | 0.9697 | 0.9547 | 65 |
| malignant | 0.7419 | 0.9529 | 0.8519 | 0.7931 | 0.9101 | 0.9484 | 0.9057 | 31 |
| normal | 0.8500 | 0.9479 | 0.7727 | 0.8095 | 0.9681 | 0.9437 | 0.8817 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=63 FP=4 TN=47 FN=2
- **malignant**: TP=23 FP=4 TN=81 FN=8
- **normal**: TP=17 FP=5 TN=91 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 63 | 2 | 0 |
| malignant | 3 | 23 | 5 |
| normal | 1 | 2 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9403    0.9692    0.9545        65
   malignant     0.8519    0.7419    0.7931        31
      normal     0.7727    0.8500    0.8095        20

    accuracy                         0.8879       116
   macro avg     0.8550    0.8537    0.8524       116
weighted avg     0.8878    0.8879    0.8864       116

```