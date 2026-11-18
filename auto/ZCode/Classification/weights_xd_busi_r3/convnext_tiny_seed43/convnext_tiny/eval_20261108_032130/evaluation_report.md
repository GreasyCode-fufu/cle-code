# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r3/convnext_tiny_seed43/convnext_tiny/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8375 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9501 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8557 |
| Balanced Acc | 0.8355 |
| Macro-Sensitivity | 0.8355 |
| Macro-Specificity | 0.9157 |
| Macro-Precision | 0.8398 |
| AUC-PR (Macro) | 0.9061 |
| Cohen's Kappa | 0.7527 |
| MCC | 0.7528 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1356.0 (0.74 ms/张) |
| End-to-End FPS | 178.1 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.8462 | 0.8806 | 0.8872 | 0.8627 | 0.9325 | 0.9540 | 66 |
| malignant | 0.8125 | 0.9419 | 0.8387 | 0.8254 | 0.9310 | 0.9433 | 0.8583 | 32 |
| normal | 0.8000 | 0.9592 | 0.8000 | 0.8000 | 0.9592 | 0.9745 | 0.9060 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=8 TN=44 FN=7
- **malignant**: TP=26 FP=5 TN=81 FN=6
- **normal**: TP=16 FP=4 TN=94 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 5 | 2 |
| malignant | 4 | 26 | 2 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8806    0.8939    0.8872        66
   malignant     0.8387    0.8125    0.8254        32
      normal     0.8000    0.8000    0.8000        20

    accuracy                         0.8559       118
   macro avg     0.8398    0.8355    0.8375       118
weighted avg     0.8556    0.8559    0.8557       118

```