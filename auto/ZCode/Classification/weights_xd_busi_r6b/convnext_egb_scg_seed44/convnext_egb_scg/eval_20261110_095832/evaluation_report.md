# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/convnext_egb_scg_seed44/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8549 |
| Accuracy | 0.8729 |
| Macro-AUROC (OvR) | 0.9510 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8738 |
| Balanced Acc | 0.8688 |
| Macro-Sensitivity | 0.8688 |
| Macro-Specificity | 0.9320 |
| Macro-Precision | 0.8462 |
| AUC-PR (Macro) | 0.9117 |
| Cohen's Kappa | 0.7857 |
| MCC | 0.7869 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 666.7 (1.50 ms/张) |
| End-to-End FPS | 151.8 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.9038 | 0.9219 | 0.9077 | 0.8704 | 0.9384 | 0.9573 | 66 |
| malignant | 0.8125 | 0.9535 | 0.8667 | 0.8387 | 0.9318 | 0.9451 | 0.8878 | 32 |
| normal | 0.9000 | 0.9388 | 0.7500 | 0.8182 | 0.9787 | 0.9694 | 0.8900 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=5 TN=47 FN=7
- **malignant**: TP=26 FP=4 TN=82 FN=6
- **normal**: TP=18 FP=6 TN=92 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 4 | 3 |
| malignant | 3 | 26 | 3 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9219    0.8939    0.9077        66
   malignant     0.8667    0.8125    0.8387        32
      normal     0.7500    0.9000    0.8182        20

    accuracy                         0.8729       118
   macro avg     0.8462    0.8688    0.8549       118
weighted avg     0.8778    0.8729    0.8738       118

```