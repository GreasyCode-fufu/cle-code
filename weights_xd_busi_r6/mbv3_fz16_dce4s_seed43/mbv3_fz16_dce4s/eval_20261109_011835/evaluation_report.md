# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce4s_seed43/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8137 |
| Accuracy | 0.8534 |
| Macro-AUROC (OvR) | 0.9416 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8513 |
| Balanced Acc | 0.8042 |
| Macro-Sensitivity | 0.8042 |
| Macro-Specificity | 0.9168 |
| Macro-Precision | 0.8249 |
| AUC-PR (Macro) | 0.8914 |
| Cohen's Kappa | 0.7454 |
| MCC | 0.7462 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1692.1 (0.59 ms/张) |
| End-to-End FPS | 154.3 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.8627 | 0.8971 | 0.9173 | 0.9167 | 0.9443 | 0.9512 | 65 |
| malignant | 0.7742 | 0.9294 | 0.8000 | 0.7869 | 0.9186 | 0.9491 | 0.8889 | 31 |
| normal | 0.7000 | 0.9583 | 0.7778 | 0.7368 | 0.9388 | 0.9313 | 0.8340 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=7 TN=44 FN=4
- **malignant**: TP=24 FP=6 TN=79 FN=7
- **normal**: TP=14 FP=4 TN=92 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 4 | 0 |
| malignant | 3 | 24 | 4 |
| normal | 4 | 2 | 14 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8971    0.9385    0.9173        65
   malignant     0.8000    0.7742    0.7869        31
      normal     0.7778    0.7000    0.7368        20

    accuracy                         0.8534       116
   macro avg     0.8249    0.8042    0.8137       116
weighted avg     0.8506    0.8534    0.8513       116

```