# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/swin_tiny_vanilla_patch4_window7_224_seed42/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9826 |
| Accuracy | 0.9822 |
| Macro-AUROC (OvR) | 0.9991 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9822 |
| Balanced Acc | 0.9826 |
| Macro-Sensitivity | 0.9826 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9826 |
| AUC-PR (Macro) | 0.9983 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1551.9 (0.64 ms/张) |
| End-to-End FPS | 586.4 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9982 | 0.9959 | 0.9959 | 0.9982 | 0.9999 | 0.9999 | 244 |
| NORMAL | 0.9779 | 0.9883 | 0.9779 | 0.9779 | 0.9883 | 0.9989 | 0.9978 | 271 |
| PNEUMONIA | 0.9741 | 0.9864 | 0.9741 | 0.9741 | 0.9864 | 0.9984 | 0.9971 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=540 FN=1
- **NORMAL**: TP=265 FP=6 TN=508 FN=6
- **PNEUMONIA**: TP=263 FP=7 TN=508 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 265 | 6 |
| PNEUMONIA | 1 | 6 | 263 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9779    0.9779    0.9779       271
   PNEUMONIA     0.9741    0.9741    0.9741       270

    accuracy                         0.9822       785
   macro avg     0.9826    0.9826    0.9826       785
weighted avg     0.9822    0.9822    0.9822       785

```