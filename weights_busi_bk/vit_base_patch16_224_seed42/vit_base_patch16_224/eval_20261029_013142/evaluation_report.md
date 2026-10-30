# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/vit_base_patch16_224_seed42/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8431 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9435 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8698 |
| Balanced Acc | 0.8435 |
| Macro-Sensitivity | 0.8435 |
| Macro-Specificity | 0.9256 |
| Macro-Precision | 0.8521 |
| AUC-PR (Macro) | 0.8690 |
| Cohen's Kappa | 0.7770 |
| MCC | 0.7797 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 841.8 (1.19 ms/张) |
| End-to-End FPS | 158.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.8627 | 0.8971 | 0.9173 | 0.9167 | 0.9587 | 0.9408 | 65 |
| malignant | 0.7419 | 0.9765 | 0.9200 | 0.8214 | 0.9121 | 0.9389 | 0.9031 | 31 |
| normal | 0.8500 | 0.9375 | 0.7391 | 0.7907 | 0.9677 | 0.9328 | 0.7632 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=7 TN=44 FN=4
- **malignant**: TP=23 FP=2 TN=83 FN=8
- **normal**: TP=17 FP=6 TN=90 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 2 | 2 |
| malignant | 4 | 23 | 4 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8971    0.9385    0.9173        65
   malignant     0.9200    0.7419    0.8214        31
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.8707       116
   macro avg     0.8521    0.8435    0.8431       116
weighted avg     0.8760    0.8707    0.8698       116

```