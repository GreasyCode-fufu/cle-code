# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/vit_base_patch16_224_seed42/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9776 |
| Accuracy | 0.9771 |
| Macro-AUROC (OvR) | 0.9980 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9771 |
| Balanced Acc | 0.9774 |
| Macro-Sensitivity | 0.9774 |
| Macro-Specificity | 0.9883 |
| Macro-Precision | 0.9780 |
| AUC-PR (Macro) | 0.9970 |
| Cohen's Kappa | 0.9656 |
| MCC | 0.9656 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 847.0 (1.18 ms/张) |
| End-to-End FPS | 442.9 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9877 | 1.0000 | 1.0000 | 0.9938 | 0.9945 | 0.9974 | 0.9976 | 244 |
| NORMAL | 0.9815 | 0.9805 | 0.9638 | 0.9726 | 0.9902 | 0.9986 | 0.9976 | 271 |
| PNEUMONIA | 0.9630 | 0.9845 | 0.9701 | 0.9665 | 0.9807 | 0.9979 | 0.9960 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=241 FP=0 TN=541 FN=3
- **NORMAL**: TP=266 FP=10 TN=504 FN=5
- **PNEUMONIA**: TP=260 FP=8 TN=507 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 241 | 0 | 3 |
| NORMAL | 0 | 266 | 5 |
| PNEUMONIA | 0 | 10 | 260 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9877    0.9938       244
      NORMAL     0.9638    0.9815    0.9726       271
   PNEUMONIA     0.9701    0.9630    0.9665       270

    accuracy                         0.9771       785
   macro avg     0.9780    0.9774    0.9776       785
weighted avg     0.9772    0.9771    0.9771       785

```