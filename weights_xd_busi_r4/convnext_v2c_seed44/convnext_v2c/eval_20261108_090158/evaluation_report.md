# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r4/convnext_v2c_seed44/convnext_v2c/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8463 |
| Accuracy | 0.8793 |
| Macro-AUROC (OvR) | 0.9452 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8771 |
| Balanced Acc | 0.8430 |
| Macro-Sensitivity | 0.8430 |
| Macro-Specificity | 0.9295 |
| Macro-Precision | 0.8652 |
| AUC-PR (Macro) | 0.8768 |
| Cohen's Kappa | 0.7900 |
| MCC | 0.7953 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1225.4 (0.82 ms/张) |
| End-to-End FPS | 163.7 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9692 | 0.8627 | 0.9000 | 0.9333 | 0.9565 | 0.9729 | 0.9731 | 65 |
| malignant | 0.7097 | 0.9882 | 0.9565 | 0.8148 | 0.9032 | 0.9366 | 0.9002 | 31 |
| normal | 0.8500 | 0.9375 | 0.7391 | 0.7907 | 0.9677 | 0.9260 | 0.7572 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=63 FP=7 TN=44 FN=2
- **malignant**: TP=22 FP=1 TN=84 FN=9
- **normal**: TP=17 FP=6 TN=90 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 63 | 1 | 1 |
| malignant | 4 | 22 | 5 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9000    0.9692    0.9333        65
   malignant     0.9565    0.7097    0.8148        31
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.8793       116
   macro avg     0.8652    0.8430    0.8463       116
weighted avg     0.8874    0.8793    0.8771       116

```