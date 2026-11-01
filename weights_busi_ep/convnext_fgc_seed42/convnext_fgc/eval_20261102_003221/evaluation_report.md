# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_fgc_seed42/convnext_fgc/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8396 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9562 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8569 |
| Balanced Acc | 0.8533 |
| Macro-Sensitivity | 0.8533 |
| Macro-Specificity | 0.9197 |
| Macro-Precision | 0.8351 |
| AUC-PR (Macro) | 0.9323 |
| Cohen's Kappa | 0.7565 |
| MCC | 0.7585 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 363.3 (2.75 ms/张) |
| End-to-End FPS | 127.9 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8654 | 0.8923 | 0.8855 | 0.8491 | 0.9341 | 0.9561 | 66 |
| malignant | 0.7812 | 0.9651 | 0.8929 | 0.8333 | 0.9222 | 0.9611 | 0.9170 | 32 |
| normal | 0.9000 | 0.9286 | 0.7200 | 0.8000 | 0.9785 | 0.9735 | 0.9238 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=7 TN=45 FN=8
- **malignant**: TP=25 FP=3 TN=83 FN=7
- **normal**: TP=18 FP=7 TN=91 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 3 | 5 |
| malignant | 5 | 25 | 2 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8923    0.8788    0.8855        66
   malignant     0.8929    0.7812    0.8333        32
      normal     0.7200    0.9000    0.8000        20

    accuracy                         0.8559       118
   macro avg     0.8351    0.8533    0.8396       118
weighted avg     0.8633    0.8559    0.8569       118

```