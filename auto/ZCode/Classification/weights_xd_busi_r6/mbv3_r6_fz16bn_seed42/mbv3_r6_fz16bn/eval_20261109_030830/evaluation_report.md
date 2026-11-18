# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz16bn_seed42/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7740 |
| Accuracy | 0.8051 |
| Macro-AUROC (OvR) | 0.9464 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8087 |
| Balanced Acc | 0.7891 |
| Macro-Sensitivity | 0.7891 |
| Macro-Specificity | 0.8958 |
| Macro-Precision | 0.7731 |
| AUC-PR (Macro) | 0.8984 |
| Cohen's Kappa | 0.6729 |
| MCC | 0.6761 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1080.1 (0.93 ms/张) |
| End-to-End FPS | 55.0 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.8462 | 0.8750 | 0.8615 | 0.8148 | 0.9376 | 0.9570 | 66 |
| malignant | 0.7188 | 0.9535 | 0.8519 | 0.7797 | 0.9011 | 0.9535 | 0.8831 | 32 |
| normal | 0.8000 | 0.8878 | 0.5926 | 0.6809 | 0.9560 | 0.9480 | 0.8550 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=8 TN=44 FN=10
- **malignant**: TP=23 FP=4 TN=82 FN=9
- **normal**: TP=16 FP=11 TN=87 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 4 | 6 |
| malignant | 4 | 23 | 5 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8750    0.8485    0.8615        66
   malignant     0.8519    0.7188    0.7797        32
      normal     0.5926    0.8000    0.6809        20

    accuracy                         0.8051       118
   macro avg     0.7731    0.7891    0.7740       118
weighted avg     0.8209    0.8051    0.8087       118

```