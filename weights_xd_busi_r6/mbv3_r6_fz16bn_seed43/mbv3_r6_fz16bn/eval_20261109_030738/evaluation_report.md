# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz16bn_seed43/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7919 |
| Accuracy | 0.8220 |
| Macro-AUROC (OvR) | 0.9512 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8215 |
| Balanced Acc | 0.8117 |
| Macro-Sensitivity | 0.8117 |
| Macro-Specificity | 0.9051 |
| Macro-Precision | 0.7889 |
| AUC-PR (Macro) | 0.9022 |
| Cohen's Kappa | 0.7001 |
| MCC | 0.7038 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1320.6 (0.76 ms/张) |
| End-to-End FPS | 53.8 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8654 | 0.8923 | 0.8855 | 0.8491 | 0.9490 | 0.9665 | 66 |
| malignant | 0.6562 | 0.9419 | 0.8077 | 0.7241 | 0.8804 | 0.9379 | 0.8596 | 32 |
| normal | 0.9000 | 0.9082 | 0.6667 | 0.7660 | 0.9780 | 0.9668 | 0.8805 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=7 TN=45 FN=8
- **malignant**: TP=21 FP=5 TN=81 FN=11
- **normal**: TP=18 FP=9 TN=89 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 5 | 3 |
| malignant | 5 | 21 | 6 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8923    0.8788    0.8855        66
   malignant     0.8077    0.6562    0.7241        32
      normal     0.6667    0.9000    0.7660        20

    accuracy                         0.8220       118
   macro avg     0.7889    0.8117    0.7919       118
weighted avg     0.8311    0.8220    0.8215       118

```