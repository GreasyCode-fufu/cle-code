# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_svd_seed43/convnext_svd/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8119 |
| Accuracy | 0.8534 |
| Macro-AUROC (OvR) | 0.9454 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8486 |
| Balanced Acc | 0.7930 |
| Macro-Sensitivity | 0.7930 |
| Macro-Specificity | 0.9090 |
| Macro-Precision | 0.8403 |
| AUC-PR (Macro) | 0.8808 |
| Cohen's Kappa | 0.7397 |
| MCC | 0.7456 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 350.2 (2.86 ms/张) |
| End-to-End FPS | 123.0 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9692 | 0.8039 | 0.8630 | 0.9130 | 0.9535 | 0.9614 | 0.9566 | 65 |
| malignant | 0.7097 | 0.9647 | 0.8800 | 0.7857 | 0.9011 | 0.9560 | 0.9077 | 31 |
| normal | 0.7000 | 0.9583 | 0.7778 | 0.7368 | 0.9388 | 0.9187 | 0.7781 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=63 FP=10 TN=41 FN=2
- **malignant**: TP=22 FP=3 TN=82 FN=9
- **normal**: TP=14 FP=4 TN=92 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 63 | 2 | 0 |
| malignant | 5 | 22 | 4 |
| normal | 5 | 1 | 14 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8630    0.9692    0.9130        65
   malignant     0.8800    0.7097    0.7857        31
      normal     0.7778    0.7000    0.7368        20

    accuracy                         0.8534       116
   macro avg     0.8403    0.7930    0.8119       116
weighted avg     0.8529    0.8534    0.8486       116

```