# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz16_seed42/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7686 |
| Accuracy | 0.7931 |
| Macro-AUROC (OvR) | 0.9232 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7978 |
| Balanced Acc | 0.8032 |
| Macro-Sensitivity | 0.8032 |
| Macro-Specificity | 0.8982 |
| Macro-Precision | 0.7570 |
| AUC-PR (Macro) | 0.8597 |
| Cohen's Kappa | 0.6626 |
| MCC | 0.6699 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1497.3 (0.67 ms/张) |
| End-to-End FPS | 165.8 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8000 | 0.9020 | 0.9123 | 0.8525 | 0.7797 | 0.9336 | 0.9409 | 65 |
| malignant | 0.7097 | 0.9176 | 0.7586 | 0.7333 | 0.8966 | 0.9021 | 0.8431 | 31 |
| normal | 0.9000 | 0.8750 | 0.6000 | 0.7200 | 0.9767 | 0.9339 | 0.7952 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=52 FP=5 TN=46 FN=13
- **malignant**: TP=22 FP=7 TN=78 FN=9
- **normal**: TP=18 FP=12 TN=84 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 52 | 7 | 6 |
| malignant | 3 | 22 | 6 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9123    0.8000    0.8525        65
   malignant     0.7586    0.7097    0.7333        31
      normal     0.6000    0.9000    0.7200        20

    accuracy                         0.7931       116
   macro avg     0.7570    0.8032    0.7686       116
weighted avg     0.8174    0.7931    0.7978       116

```