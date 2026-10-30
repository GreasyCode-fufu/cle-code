# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/vit_base_patch16_224_seed42/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8644 |
| Accuracy | 0.8814 |
| Macro-AUROC (OvR) | 0.9545 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8832 |
| Balanced Acc | 0.8846 |
| Macro-Sensitivity | 0.8846 |
| Macro-Specificity | 0.9410 |
| Macro-Precision | 0.8498 |
| AUC-PR (Macro) | 0.9059 |
| Cohen's Kappa | 0.8024 |
| MCC | 0.8047 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 846.5 (1.18 ms/张) |
| End-to-End FPS | 164.9 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.9423 | 0.9508 | 0.9134 | 0.8596 | 0.9362 | 0.9621 | 66 |
| malignant | 0.8750 | 0.9419 | 0.8485 | 0.8615 | 0.9529 | 0.9528 | 0.8672 | 32 |
| normal | 0.9000 | 0.9388 | 0.7500 | 0.8182 | 0.9787 | 0.9745 | 0.8886 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=3 TN=49 FN=8
- **malignant**: TP=28 FP=5 TN=81 FN=4
- **normal**: TP=18 FP=6 TN=92 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 5 | 3 |
| malignant | 1 | 28 | 3 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9508    0.8788    0.9134        66
   malignant     0.8485    0.8750    0.8615        32
      normal     0.7500    0.9000    0.8182        20

    accuracy                         0.8814       118
   macro avg     0.8498    0.8846    0.8644       118
weighted avg     0.8890    0.8814    0.8832       118

```