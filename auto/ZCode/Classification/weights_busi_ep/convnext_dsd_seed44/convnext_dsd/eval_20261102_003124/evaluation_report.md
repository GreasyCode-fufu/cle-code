# convnext_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_dsd_seed44/convnext_dsd/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8395 |
| Accuracy | 0.8707 |
| Macro-AUROC (OvR) | 0.9528 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8706 |
| Balanced Acc | 0.8435 |
| Macro-Sensitivity | 0.8435 |
| Macro-Specificity | 0.9286 |
| Macro-Precision | 0.8463 |
| AUC-PR (Macro) | 0.8936 |
| Cohen's Kappa | 0.7782 |
| MCC | 0.7810 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 420.2 (2.38 ms/张) |
| End-to-End FPS | 129.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.8824 | 0.9104 | 0.9242 | 0.9184 | 0.9674 | 0.9504 | 65 |
| malignant | 0.7419 | 0.9765 | 0.9200 | 0.8214 | 0.9121 | 0.9681 | 0.9353 | 31 |
| normal | 0.8500 | 0.9271 | 0.7083 | 0.7727 | 0.9674 | 0.9229 | 0.7951 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=6 TN=45 FN=4
- **malignant**: TP=23 FP=2 TN=83 FN=8
- **normal**: TP=17 FP=7 TN=89 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 2 | 2 |
| malignant | 3 | 23 | 5 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9104    0.9385    0.9242        65
   malignant     0.9200    0.7419    0.8214        31
      normal     0.7083    0.8500    0.7727        20

    accuracy                         0.8707       116
   macro avg     0.8463    0.8435    0.8395       116
weighted avg     0.8782    0.8707    0.8706       116

```