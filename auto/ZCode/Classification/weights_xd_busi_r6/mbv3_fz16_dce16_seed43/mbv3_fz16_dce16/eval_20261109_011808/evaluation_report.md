# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce16_seed43/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8507 |
| Accuracy | 0.8644 |
| Macro-AUROC (OvR) | 0.9457 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8637 |
| Balanced Acc | 0.8468 |
| Macro-Sensitivity | 0.8468 |
| Macro-Specificity | 0.9191 |
| Macro-Precision | 0.8552 |
| AUC-PR (Macro) | 0.8927 |
| Cohen's Kappa | 0.7662 |
| MCC | 0.7666 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 1720.4 (0.58 ms/张) |
| End-to-End FPS | 158.8 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9091 | 0.8462 | 0.8824 | 0.8955 | 0.8800 | 0.9343 | 0.9524 | 66 |
| malignant | 0.7812 | 0.9419 | 0.8333 | 0.8065 | 0.9205 | 0.9488 | 0.8632 | 32 |
| normal | 0.8500 | 0.9694 | 0.8500 | 0.8500 | 0.9694 | 0.9541 | 0.8624 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=60 FP=8 TN=44 FN=6
- **malignant**: TP=25 FP=5 TN=81 FN=7
- **normal**: TP=17 FP=3 TN=95 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 60 | 5 | 1 |
| malignant | 5 | 25 | 2 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8824    0.9091    0.8955        66
   malignant     0.8333    0.7812    0.8065        32
      normal     0.8500    0.8500    0.8500        20

    accuracy                         0.8644       118
   macro avg     0.8552    0.8468    0.8507       118
weighted avg     0.8636    0.8644    0.8637       118

```