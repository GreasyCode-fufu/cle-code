# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/convnext_egb_scg_seed42/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8673 |
| Accuracy | 0.8879 |
| Macro-AUROC (OvR) | 0.9563 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8870 |
| Balanced Acc | 0.8709 |
| Macro-Sensitivity | 0.8709 |
| Macro-Specificity | 0.9377 |
| Macro-Precision | 0.8665 |
| AUC-PR (Macro) | 0.9189 |
| Cohen's Kappa | 0.8081 |
| MCC | 0.8088 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 542.0 (1.84 ms/张) |
| End-to-End FPS | 139.7 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.9020 | 0.9242 | 0.9313 | 0.9200 | 0.9602 | 0.9562 | 65 |
| malignant | 0.7742 | 0.9529 | 0.8571 | 0.8136 | 0.9205 | 0.9552 | 0.9230 | 31 |
| normal | 0.9000 | 0.9583 | 0.8182 | 0.8571 | 0.9787 | 0.9536 | 0.8774 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=5 TN=46 FN=4
- **malignant**: TP=24 FP=4 TN=81 FN=7
- **normal**: TP=18 FP=4 TN=92 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 4 | 0 |
| malignant | 3 | 24 | 4 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9242    0.9385    0.9313        65
   malignant     0.8571    0.7742    0.8136        31
      normal     0.8182    0.9000    0.8571        20

    accuracy                         0.8879       116
   macro avg     0.8665    0.8709    0.8673       116
weighted avg     0.8880    0.8879    0.8870       116

```