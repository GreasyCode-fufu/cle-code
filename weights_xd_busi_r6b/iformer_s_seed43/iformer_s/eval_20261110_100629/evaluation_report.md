# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/iformer_s_seed43/iformer_s/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7909 |
| Accuracy | 0.8190 |
| Macro-AUROC (OvR) | 0.9151 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8225 |
| Balanced Acc | 0.8127 |
| Macro-Sensitivity | 0.8127 |
| Macro-Specificity | 0.9108 |
| Macro-Precision | 0.7770 |
| AUC-PR (Macro) | 0.8061 |
| Cohen's Kappa | 0.7003 |
| MCC | 0.7033 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 2840.3 (0.35 ms/张) |
| End-to-End FPS | 174.9 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8462 | 0.9216 | 0.9322 | 0.8871 | 0.8246 | 0.9234 | 0.9376 | 65 |
| malignant | 0.7419 | 0.8941 | 0.7188 | 0.7302 | 0.9048 | 0.8964 | 0.7953 | 31 |
| normal | 0.8500 | 0.9167 | 0.6800 | 0.7556 | 0.9670 | 0.9255 | 0.6853 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=55 FP=4 TN=47 FN=10
- **malignant**: TP=23 FP=9 TN=76 FN=8
- **normal**: TP=17 FP=8 TN=88 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 55 | 7 | 3 |
| malignant | 3 | 23 | 5 |
| normal | 1 | 2 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9322    0.8462    0.8871        65
   malignant     0.7188    0.7419    0.7302        31
      normal     0.6800    0.8500    0.7556        20

    accuracy                         0.8190       116
   macro avg     0.7770    0.8127    0.7909       116
weighted avg     0.8317    0.8190    0.8225       116

```