# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/convnext_small_seed42/convnext_small/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8499 |
| Accuracy | 0.8793 |
| Macro-AUROC (OvR) | 0.9551 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8793 |
| Balanced Acc | 0.8542 |
| Macro-Sensitivity | 0.8542 |
| Macro-Specificity | 0.9347 |
| Macro-Precision | 0.8508 |
| AUC-PR (Macro) | 0.9103 |
| Cohen's Kappa | 0.7936 |
| MCC | 0.7949 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 907.7 (1.10 ms/张) |
| End-to-End FPS | 161.3 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.9020 | 0.9242 | 0.9313 | 0.9200 | 0.9638 | 0.9700 | 65 |
| malignant | 0.7742 | 0.9647 | 0.8889 | 0.8276 | 0.9213 | 0.9397 | 0.8846 | 31 |
| normal | 0.8500 | 0.9375 | 0.7391 | 0.7907 | 0.9677 | 0.9620 | 0.8764 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=5 TN=46 FN=4
- **malignant**: TP=24 FP=3 TN=82 FN=7
- **normal**: TP=17 FP=6 TN=90 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 3 | 1 |
| malignant | 2 | 24 | 5 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9242    0.9385    0.9313        65
   malignant     0.8889    0.7742    0.8276        31
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.8793       116
   macro avg     0.8508    0.8542    0.8499       116
weighted avg     0.8829    0.8793    0.8793       116

```