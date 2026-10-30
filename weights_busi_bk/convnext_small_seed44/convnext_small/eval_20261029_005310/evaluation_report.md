# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/convnext_small_seed44/convnext_small/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8106 |
| Accuracy | 0.8305 |
| Macro-AUROC (OvR) | 0.9340 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8322 |
| Balanced Acc | 0.8328 |
| Macro-Sensitivity | 0.8328 |
| Macro-Specificity | 0.9131 |
| Macro-Precision | 0.7960 |
| AUC-PR (Macro) | 0.8747 |
| Cohen's Kappa | 0.7182 |
| MCC | 0.7205 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 910.1 (1.10 ms/张) |
| End-to-End FPS | 168.1 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.9038 | 0.9180 | 0.8819 | 0.8246 | 0.9151 | 0.9447 | 66 |
| malignant | 0.7500 | 0.9070 | 0.7500 | 0.7500 | 0.9070 | 0.9130 | 0.7606 | 32 |
| normal | 0.9000 | 0.9286 | 0.7200 | 0.8000 | 0.9785 | 0.9740 | 0.9188 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=5 TN=47 FN=10
- **malignant**: TP=24 FP=8 TN=78 FN=8
- **normal**: TP=18 FP=7 TN=91 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 8 | 2 |
| malignant | 3 | 24 | 5 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9180    0.8485    0.8819        66
   malignant     0.7500    0.7500    0.7500        32
      normal     0.7200    0.9000    0.8000        20

    accuracy                         0.8305       118
   macro avg     0.7960    0.8328    0.8106       118
weighted avg     0.8389    0.8305    0.8322       118

```