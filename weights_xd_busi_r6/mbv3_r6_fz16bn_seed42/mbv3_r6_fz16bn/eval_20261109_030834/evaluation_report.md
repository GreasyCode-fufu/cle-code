# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz16bn_seed42/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8162 |
| Accuracy | 0.8448 |
| Macro-AUROC (OvR) | 0.9353 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8489 |
| Balanced Acc | 0.8337 |
| Macro-Sensitivity | 0.8337 |
| Macro-Specificity | 0.9208 |
| Macro-Precision | 0.8126 |
| AUC-PR (Macro) | 0.8827 |
| Cohen's Kappa | 0.7406 |
| MCC | 0.7441 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1221.2 (0.82 ms/张) |
| End-to-End FPS | 134.5 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8769 | 0.9020 | 0.9194 | 0.8976 | 0.8519 | 0.9339 | 0.9424 | 65 |
| malignant | 0.7742 | 0.9647 | 0.8889 | 0.8276 | 0.9213 | 0.9313 | 0.8793 | 31 |
| normal | 0.8500 | 0.8958 | 0.6296 | 0.7234 | 0.9663 | 0.9406 | 0.8265 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=5 TN=46 FN=8
- **malignant**: TP=24 FP=3 TN=82 FN=7
- **normal**: TP=17 FP=10 TN=86 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 3 | 5 |
| malignant | 2 | 24 | 5 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9194    0.8769    0.8976        65
   malignant     0.8889    0.7742    0.8276        31
      normal     0.6296    0.8500    0.7234        20

    accuracy                         0.8448       116
   macro avg     0.8126    0.8337    0.8162       116
weighted avg     0.8613    0.8448    0.8489       116

```