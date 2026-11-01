# convnext_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_dsd_seed42/convnext_dsd/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7993 |
| Accuracy | 0.8220 |
| Macro-AUROC (OvR) | 0.9450 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8236 |
| Balanced Acc | 0.8162 |
| Macro-Sensitivity | 0.8162 |
| Macro-Specificity | 0.9047 |
| Macro-Precision | 0.7896 |
| AUC-PR (Macro) | 0.8950 |
| Cohen's Kappa | 0.7017 |
| MCC | 0.7034 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 408.9 (2.45 ms/张) |
| End-to-End FPS | 134.0 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.8654 | 0.8889 | 0.8682 | 0.8182 | 0.9328 | 0.9533 | 66 |
| malignant | 0.7500 | 0.9302 | 0.8000 | 0.7742 | 0.9091 | 0.9328 | 0.8432 | 32 |
| normal | 0.8500 | 0.9184 | 0.6800 | 0.7556 | 0.9677 | 0.9694 | 0.8884 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=7 TN=45 FN=10
- **malignant**: TP=24 FP=6 TN=80 FN=8
- **normal**: TP=17 FP=8 TN=90 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 5 | 5 |
| malignant | 5 | 24 | 3 |
| normal | 2 | 1 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8889    0.8485    0.8682        66
   malignant     0.8000    0.7500    0.7742        32
      normal     0.6800    0.8500    0.7556        20

    accuracy                         0.8220       118
   macro avg     0.7896    0.8162    0.7993       118
weighted avg     0.8294    0.8220    0.8236       118

```