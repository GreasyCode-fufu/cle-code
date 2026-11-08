# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz16_seed44/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7955 |
| Accuracy | 0.8276 |
| Macro-AUROC (OvR) | 0.9255 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8308 |
| Balanced Acc | 0.8063 |
| Macro-Sensitivity | 0.8063 |
| Macro-Specificity | 0.9078 |
| Macro-Precision | 0.7969 |
| AUC-PR (Macro) | 0.8646 |
| Cohen's Kappa | 0.7089 |
| MCC | 0.7117 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1710.4 (0.58 ms/张) |
| End-to-End FPS | 175.2 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8769 | 0.8627 | 0.8906 | 0.8837 | 0.8462 | 0.9436 | 0.9489 | 65 |
| malignant | 0.7419 | 0.9647 | 0.8846 | 0.8070 | 0.9111 | 0.9044 | 0.8618 | 31 |
| normal | 0.8000 | 0.8958 | 0.6154 | 0.6957 | 0.9556 | 0.9286 | 0.7830 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=7 TN=44 FN=8
- **malignant**: TP=23 FP=3 TN=82 FN=8
- **normal**: TP=16 FP=10 TN=86 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 3 | 5 |
| malignant | 3 | 23 | 5 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8906    0.8769    0.8837        65
   malignant     0.8846    0.7419    0.8070        31
      normal     0.6154    0.8000    0.6957        20

    accuracy                         0.8276       116
   macro avg     0.7969    0.8063    0.7955       116
weighted avg     0.8416    0.8276    0.8308       116

```