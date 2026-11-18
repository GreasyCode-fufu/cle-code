# mbv3_fz16_dce16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce16_seed43/mbv3_fz16_dce16/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8095 |
| Accuracy | 0.8534 |
| Macro-AUROC (OvR) | 0.9365 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8487 |
| Balanced Acc | 0.7930 |
| Macro-Sensitivity | 0.7930 |
| Macro-Specificity | 0.9116 |
| Macro-Precision | 0.8330 |
| AUC-PR (Macro) | 0.8605 |
| Cohen's Kappa | 0.7409 |
| MCC | 0.7453 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 1677.8 (0.60 ms/张) |
| End-to-End FPS | 144.1 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9692 | 0.8235 | 0.8750 | 0.9197 | 0.9545 | 0.9400 | 0.9353 | 65 |
| malignant | 0.7097 | 0.9529 | 0.8462 | 0.7719 | 0.9000 | 0.9381 | 0.8775 | 31 |
| normal | 0.7000 | 0.9583 | 0.7778 | 0.7368 | 0.9388 | 0.9313 | 0.7687 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=63 FP=9 TN=42 FN=2
- **malignant**: TP=22 FP=4 TN=81 FN=9
- **normal**: TP=14 FP=4 TN=92 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 63 | 2 | 0 |
| malignant | 5 | 22 | 4 |
| normal | 4 | 2 | 14 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8750    0.9692    0.9197        65
   malignant     0.8462    0.7097    0.7719        31
      normal     0.7778    0.7000    0.7368        20

    accuracy                         0.8534       116
   macro avg     0.8330    0.7930    0.8095       116
weighted avg     0.8505    0.8534    0.8487       116

```