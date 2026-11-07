# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r3/convnext_tiny_seed42/convnext_tiny/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8589 |
| Accuracy | 0.8793 |
| Macro-AUROC (OvR) | 0.9585 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8799 |
| Balanced Acc | 0.8655 |
| Macro-Sensitivity | 0.8655 |
| Macro-Specificity | 0.9343 |
| Macro-Precision | 0.8538 |
| AUC-PR (Macro) | 0.9295 |
| Cohen's Kappa | 0.7951 |
| MCC | 0.7954 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 609.9 (1.64 ms/张) |
| End-to-End FPS | 143.1 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9077 | 0.9020 | 0.9219 | 0.9147 | 0.8846 | 0.9643 | 0.9670 | 65 |
| malignant | 0.8387 | 0.9529 | 0.8667 | 0.8525 | 0.9419 | 0.9613 | 0.9258 | 31 |
| normal | 0.8500 | 0.9479 | 0.7727 | 0.8095 | 0.9681 | 0.9500 | 0.8957 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=5 TN=46 FN=6
- **malignant**: TP=26 FP=4 TN=81 FN=5
- **normal**: TP=17 FP=5 TN=91 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 4 | 2 |
| malignant | 2 | 26 | 3 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9219    0.9077    0.9147        65
   malignant     0.8667    0.8387    0.8525        31
      normal     0.7727    0.8500    0.8095        20

    accuracy                         0.8793       116
   macro avg     0.8538    0.8655    0.8589       116
weighted avg     0.8814    0.8793    0.8799       116

```