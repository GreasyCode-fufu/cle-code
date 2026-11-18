# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce4m_seed42/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7889 |
| Accuracy | 0.8276 |
| Macro-AUROC (OvR) | 0.9295 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8268 |
| Balanced Acc | 0.7891 |
| Macro-Sensitivity | 0.7891 |
| Macro-Specificity | 0.9038 |
| Macro-Precision | 0.7924 |
| AUC-PR (Macro) | 0.8768 |
| Cohen's Kappa | 0.7035 |
| MCC | 0.7046 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1653.7 (0.60 ms/张) |
| End-to-End FPS | 152.5 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9077 | 0.8431 | 0.8806 | 0.8939 | 0.8776 | 0.9409 | 0.9390 | 65 |
| malignant | 0.7097 | 0.9412 | 0.8148 | 0.7586 | 0.8989 | 0.9214 | 0.8632 | 31 |
| normal | 0.7500 | 0.9271 | 0.6818 | 0.7143 | 0.9468 | 0.9260 | 0.8282 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=8 TN=43 FN=6
- **malignant**: TP=22 FP=5 TN=80 FN=9
- **normal**: TP=15 FP=7 TN=89 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 4 | 2 |
| malignant | 4 | 22 | 5 |
| normal | 4 | 1 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8806    0.9077    0.8939        65
   malignant     0.8148    0.7097    0.7586        31
      normal     0.6818    0.7500    0.7143        20

    accuracy                         0.8276       116
   macro avg     0.7924    0.7891    0.7889       116
weighted avg     0.8287    0.8276    0.8268       116

```