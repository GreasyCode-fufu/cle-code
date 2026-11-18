# fast_t0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/fast_t0_seed44/fast_t0/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7774 |
| Accuracy | 0.8017 |
| Macro-AUROC (OvR) | 0.9346 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8031 |
| Balanced Acc | 0.7850 |
| Macro-Sensitivity | 0.7850 |
| Macro-Specificity | 0.8916 |
| Macro-Precision | 0.7708 |
| AUC-PR (Macro) | 0.8705 |
| Cohen's Kappa | 0.6658 |
| MCC | 0.6664 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 2749.8 (0.36 ms/张) |
| End-to-End FPS | 182.9 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8308 | 0.8431 | 0.8710 | 0.8504 | 0.7963 | 0.9303 | 0.9416 | 65 |
| malignant | 0.7742 | 0.8941 | 0.7273 | 0.7500 | 0.9157 | 0.9438 | 0.8724 | 31 |
| normal | 0.7500 | 0.9375 | 0.7143 | 0.7317 | 0.9474 | 0.9297 | 0.7975 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=54 FP=8 TN=43 FN=11
- **malignant**: TP=24 FP=9 TN=76 FN=7
- **normal**: TP=15 FP=6 TN=90 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 54 | 9 | 2 |
| malignant | 3 | 24 | 4 |
| normal | 5 | 0 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8710    0.8308    0.8504        65
   malignant     0.7273    0.7742    0.7500        31
      normal     0.7143    0.7500    0.7317        20

    accuracy                         0.8017       116
   macro avg     0.7708    0.7850    0.7774       116
weighted avg     0.8056    0.8017    0.8031       116

```