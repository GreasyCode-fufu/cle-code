# convnext_dph 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_dph_seed43/convnext_dph/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8387 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9491 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8576 |
| Balanced Acc | 0.8587 |
| Macro-Sensitivity | 0.8587 |
| Macro-Specificity | 0.9248 |
| Macro-Precision | 0.8260 |
| AUC-PR (Macro) | 0.8822 |
| Cohen's Kappa | 0.7595 |
| MCC | 0.7616 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 415.1 (2.41 ms/张) |
| End-to-End FPS | 134.2 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8636 | 0.9038 | 0.9194 | 0.8906 | 0.8393 | 0.9315 | 0.9599 | 66 |
| malignant | 0.8125 | 0.9419 | 0.8387 | 0.8254 | 0.9310 | 0.9422 | 0.8590 | 32 |
| normal | 0.9000 | 0.9286 | 0.7200 | 0.8000 | 0.9785 | 0.9735 | 0.8277 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=5 TN=47 FN=9
- **malignant**: TP=26 FP=5 TN=81 FN=6
- **normal**: TP=18 FP=7 TN=91 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 5 | 4 |
| malignant | 3 | 26 | 3 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9194    0.8636    0.8906        66
   malignant     0.8387    0.8125    0.8254        32
      normal     0.7200    0.9000    0.8000        20

    accuracy                         0.8559       118
   macro avg     0.8260    0.8587    0.8387       118
weighted avg     0.8637    0.8559    0.8576       118

```