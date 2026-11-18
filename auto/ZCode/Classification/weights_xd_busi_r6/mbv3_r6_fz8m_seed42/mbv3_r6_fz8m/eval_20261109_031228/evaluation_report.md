# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz8m_seed42/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8341 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9556 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8573 |
| Balanced Acc | 0.8471 |
| Macro-Sensitivity | 0.8471 |
| Macro-Specificity | 0.9243 |
| Macro-Precision | 0.8241 |
| AUC-PR (Macro) | 0.9155 |
| Cohen's Kappa | 0.7578 |
| MCC | 0.7587 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1246.5 (0.80 ms/张) |
| End-to-End FPS | 25.4 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.9038 | 0.9206 | 0.8992 | 0.8545 | 0.9513 | 0.9612 | 66 |
| malignant | 0.8125 | 0.9302 | 0.8125 | 0.8125 | 0.9302 | 0.9488 | 0.8863 | 32 |
| normal | 0.8500 | 0.9388 | 0.7391 | 0.7907 | 0.9684 | 0.9668 | 0.8990 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=5 TN=47 FN=8
- **malignant**: TP=26 FP=6 TN=80 FN=6
- **normal**: TP=17 FP=6 TN=92 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 6 | 2 |
| malignant | 2 | 26 | 4 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9206    0.8788    0.8992        66
   malignant     0.8125    0.8125    0.8125        32
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.8559       118
   macro avg     0.8241    0.8471    0.8341       118
weighted avg     0.8605    0.8559    0.8573       118

```