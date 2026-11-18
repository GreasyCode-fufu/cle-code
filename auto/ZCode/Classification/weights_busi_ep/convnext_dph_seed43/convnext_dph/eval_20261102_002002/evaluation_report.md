# convnext_dph 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_dph_seed43/convnext_dph/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8661 |
| Accuracy | 0.8879 |
| Macro-AUROC (OvR) | 0.9449 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8884 |
| Balanced Acc | 0.8706 |
| Macro-Sensitivity | 0.8706 |
| Macro-Specificity | 0.9404 |
| Macro-Precision | 0.8619 |
| AUC-PR (Macro) | 0.8815 |
| Cohen's Kappa | 0.8095 |
| MCC | 0.8096 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 426.5 (2.34 ms/张) |
| End-to-End FPS | 129.9 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9231 | 0.9216 | 0.9375 | 0.9302 | 0.9038 | 0.9674 | 0.9653 | 65 |
| malignant | 0.8387 | 0.9412 | 0.8387 | 0.8387 | 0.9412 | 0.9313 | 0.9017 | 31 |
| normal | 0.8500 | 0.9583 | 0.8095 | 0.8293 | 0.9684 | 0.9359 | 0.7774 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=4 TN=47 FN=5
- **malignant**: TP=26 FP=5 TN=80 FN=5
- **normal**: TP=17 FP=4 TN=92 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 5 | 0 |
| malignant | 1 | 26 | 4 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9375    0.9231    0.9302        65
   malignant     0.8387    0.8387    0.8387        31
      normal     0.8095    0.8500    0.8293        20

    accuracy                         0.8879       116
   macro avg     0.8619    0.8706    0.8661       116
weighted avg     0.8890    0.8879    0.8884       116

```