# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/convnext_tiny_seed44/convnext_tiny/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8501 |
| Accuracy | 0.8644 |
| Macro-AUROC (OvR) | 0.9474 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8664 |
| Balanced Acc | 0.8745 |
| Macro-Sensitivity | 0.8745 |
| Macro-Specificity | 0.9312 |
| Macro-Precision | 0.8339 |
| AUC-PR (Macro) | 0.8936 |
| Cohen's Kappa | 0.7755 |
| MCC | 0.7786 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1010.7 (0.99 ms/张) |
| End-to-End FPS | 165.9 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.9231 | 0.9333 | 0.8889 | 0.8276 | 0.9285 | 0.9505 | 66 |
| malignant | 0.8750 | 0.9419 | 0.8485 | 0.8615 | 0.9529 | 0.9391 | 0.8260 | 32 |
| normal | 0.9000 | 0.9286 | 0.7200 | 0.8000 | 0.9785 | 0.9745 | 0.9042 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=4 TN=48 FN=10
- **malignant**: TP=28 FP=5 TN=81 FN=4
- **normal**: TP=18 FP=7 TN=91 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 5 | 5 |
| malignant | 2 | 28 | 2 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9333    0.8485    0.8889        66
   malignant     0.8485    0.8750    0.8615        32
      normal     0.7200    0.9000    0.8000        20

    accuracy                         0.8644       118
   macro avg     0.8339    0.8745    0.8501       118
weighted avg     0.8742    0.8644    0.8664       118

```