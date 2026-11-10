# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/lsnet_s_seed43/lsnet_s/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7354 |
| Accuracy | 0.7542 |
| Macro-AUROC (OvR) | 0.9047 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7536 |
| Balanced Acc | 0.7722 |
| Macro-Sensitivity | 0.7722 |
| Macro-Specificity | 0.8659 |
| Macro-Precision | 0.7360 |
| AUC-PR (Macro) | 0.8276 |
| Cohen's Kappa | 0.5938 |
| MCC | 0.6019 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 1661.4 (0.60 ms/张) |
| End-to-End FPS | 181.6 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.7727 | 0.7885 | 0.8226 | 0.7969 | 0.7321 | 0.8868 | 0.9019 | 66 |
| malignant | 0.5938 | 0.9419 | 0.7917 | 0.6786 | 0.8617 | 0.8810 | 0.8085 | 32 |
| normal | 0.9500 | 0.8673 | 0.5938 | 0.7308 | 0.9884 | 0.9464 | 0.7724 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=51 FP=11 TN=41 FN=15
- **malignant**: TP=19 FP=5 TN=81 FN=13
- **normal**: TP=19 FP=13 TN=85 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 51 | 5 | 10 |
| malignant | 10 | 19 | 3 |
| normal | 1 | 0 | 19 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8226    0.7727    0.7969        66
   malignant     0.7917    0.5938    0.6786        32
      normal     0.5938    0.9500    0.7308        20

    accuracy                         0.7542       118
   macro avg     0.7360    0.7722    0.7354       118
weighted avg     0.7754    0.7542    0.7536       118

```