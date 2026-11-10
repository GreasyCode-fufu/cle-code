# jumbo_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/jumbo_small_seed42/jumbo_small/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8339 |
| Accuracy | 0.8475 |
| Macro-AUROC (OvR) | 0.9580 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8504 |
| Balanced Acc | 0.8769 |
| Macro-Sensitivity | 0.8769 |
| Macro-Specificity | 0.9294 |
| Macro-Precision | 0.8139 |
| AUC-PR (Macro) | 0.9149 |
| Cohen's Kappa | 0.7528 |
| MCC | 0.7615 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.78G |
| FLOPs | 9.56G |
| Params | 64.13M |
| Pure-Inference FPS | 1130.4 (0.88 ms/张) |
| End-to-End FPS | 168.1 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8182 | 0.9615 | 0.9643 | 0.8852 | 0.8065 | 0.9420 | 0.9626 | 66 |
| malignant | 0.8125 | 0.9186 | 0.7879 | 0.8000 | 0.9294 | 0.9473 | 0.8537 | 32 |
| normal | 1.0000 | 0.9082 | 0.6897 | 0.8163 | 1.0000 | 0.9847 | 0.9285 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=54 FP=2 TN=50 FN=12
- **malignant**: TP=26 FP=7 TN=79 FN=6
- **normal**: TP=20 FP=9 TN=89 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 54 | 7 | 5 |
| malignant | 2 | 26 | 4 |
| normal | 0 | 0 | 20 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9643    0.8182    0.8852        66
   malignant     0.7879    0.8125    0.8000        32
      normal     0.6897    1.0000    0.8163        20

    accuracy                         0.8475       118
   macro avg     0.8139    0.8769    0.8339       118
weighted avg     0.8699    0.8475    0.8504       118

```