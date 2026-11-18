# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/resnet34_seed42/resnet34/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8363 |
| Accuracy | 0.8475 |
| Macro-AUROC (OvR) | 0.9429 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8479 |
| Balanced Acc | 0.8304 |
| Macro-Sensitivity | 0.8304 |
| Macro-Specificity | 0.9109 |
| Macro-Precision | 0.8441 |
| AUC-PR (Macro) | 0.9162 |
| Cohen's Kappa | 0.7384 |
| MCC | 0.7388 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 1974.1 (0.51 ms/张) |
| End-to-End FPS | 185.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8462 | 0.8788 | 0.8788 | 0.8462 | 0.9222 | 0.9345 | 66 |
| malignant | 0.8125 | 0.9070 | 0.7647 | 0.7879 | 0.9286 | 0.9219 | 0.8669 | 32 |
| normal | 0.8000 | 0.9796 | 0.8889 | 0.8421 | 0.9600 | 0.9847 | 0.9473 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=8 TN=44 FN=8
- **malignant**: TP=26 FP=8 TN=78 FN=6
- **normal**: TP=16 FP=2 TN=96 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 7 | 1 |
| malignant | 5 | 26 | 1 |
| normal | 3 | 1 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8788    0.8788    0.8788        66
   malignant     0.7647    0.8125    0.7879        32
      normal     0.8889    0.8000    0.8421        20

    accuracy                         0.8475       118
   macro avg     0.8441    0.8304    0.8363       118
weighted avg     0.8496    0.8475    0.8479       118

```