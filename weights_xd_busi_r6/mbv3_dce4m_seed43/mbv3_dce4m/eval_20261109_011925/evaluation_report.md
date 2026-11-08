# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_dce4m_seed43/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8539 |
| Accuracy | 0.8793 |
| Macro-AUROC (OvR) | 0.9519 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8793 |
| Balanced Acc | 0.8539 |
| Macro-Sensitivity | 0.8539 |
| Macro-Specificity | 0.9338 |
| Macro-Precision | 0.8539 |
| AUC-PR (Macro) | 0.9029 |
| Cohen's Kappa | 0.7936 |
| MCC | 0.7936 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1656.9 (0.60 ms/张) |
| End-to-End FPS | 153.6 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9231 | 0.9020 | 0.9231 | 0.9231 | 0.9020 | 0.9505 | 0.9538 | 65 |
| malignant | 0.8387 | 0.9412 | 0.8387 | 0.8387 | 0.9412 | 0.9567 | 0.8935 | 31 |
| normal | 0.8000 | 0.9583 | 0.8000 | 0.8000 | 0.9583 | 0.9484 | 0.8615 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=5 TN=46 FN=5
- **malignant**: TP=26 FP=5 TN=80 FN=5
- **normal**: TP=16 FP=4 TN=92 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 4 | 1 |
| malignant | 2 | 26 | 3 |
| normal | 3 | 1 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9231    0.9231    0.9231        65
   malignant     0.8387    0.8387    0.8387        31
      normal     0.8000    0.8000    0.8000        20

    accuracy                         0.8793       116
   macro avg     0.8539    0.8539    0.8539       116
weighted avg     0.8793    0.8793    0.8793       116

```