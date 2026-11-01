# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_fgc_seed43/convnext_fgc/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8382 |
| Accuracy | 0.8534 |
| Macro-AUROC (OvR) | 0.9362 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8537 |
| Balanced Acc | 0.8504 |
| Macro-Sensitivity | 0.8504 |
| Macro-Specificity | 0.9199 |
| Macro-Precision | 0.8291 |
| AUC-PR (Macro) | 0.8946 |
| Cohen's Kappa | 0.7526 |
| MCC | 0.7533 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 386.7 (2.59 ms/张) |
| End-to-End FPS | 127.1 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8769 | 0.8824 | 0.9048 | 0.8906 | 0.8491 | 0.9418 | 0.9329 | 65 |
| malignant | 0.7742 | 0.9294 | 0.8000 | 0.7869 | 0.9186 | 0.9256 | 0.8751 | 31 |
| normal | 0.9000 | 0.9479 | 0.7826 | 0.8372 | 0.9785 | 0.9411 | 0.8759 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=57 FP=6 TN=45 FN=8
- **malignant**: TP=24 FP=6 TN=79 FN=7
- **normal**: TP=18 FP=5 TN=91 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 57 | 6 | 2 |
| malignant | 4 | 24 | 3 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9048    0.8769    0.8906        65
   malignant     0.8000    0.7742    0.7869        31
      normal     0.7826    0.9000    0.8372        20

    accuracy                         0.8534       116
   macro avg     0.8291    0.8504    0.8382       116
weighted avg     0.8557    0.8534    0.8537       116

```