# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_svd_seed42/convnext_svd/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8497 |
| Accuracy | 0.8729 |
| Macro-AUROC (OvR) | 0.9531 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8723 |
| Balanced Acc | 0.8456 |
| Macro-Sensitivity | 0.8456 |
| Macro-Specificity | 0.9260 |
| Macro-Precision | 0.8546 |
| AUC-PR (Macro) | 0.9215 |
| Cohen's Kappa | 0.7808 |
| MCC | 0.7813 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 380.2 (2.63 ms/张) |
| End-to-End FPS | 129.5 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9242 | 0.8654 | 0.8971 | 0.9104 | 0.9000 | 0.9416 | 0.9544 | 66 |
| malignant | 0.8125 | 0.9535 | 0.8667 | 0.8387 | 0.9318 | 0.9626 | 0.9150 | 32 |
| normal | 0.8000 | 0.9592 | 0.8000 | 0.8000 | 0.9592 | 0.9551 | 0.8950 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=7 TN=45 FN=5
- **malignant**: TP=26 FP=4 TN=82 FN=6
- **normal**: TP=16 FP=4 TN=94 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 4 | 1 |
| malignant | 3 | 26 | 3 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8971    0.9242    0.9104        66
   malignant     0.8667    0.8125    0.8387        32
      normal     0.8000    0.8000    0.8000        20

    accuracy                         0.8729       118
   macro avg     0.8546    0.8456    0.8497       118
weighted avg     0.8724    0.8729    0.8723       118

```