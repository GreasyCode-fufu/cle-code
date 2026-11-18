# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/swin_tiny_vanilla_patch4_window7_224_seed44/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9826 |
| Accuracy | 0.9822 |
| Macro-AUROC (OvR) | 0.9990 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9822 |
| Balanced Acc | 0.9826 |
| Macro-Sensitivity | 0.9826 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9831 |
| AUC-PR (Macro) | 0.9981 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9735 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1551.5 (0.64 ms/张) |
| End-to-End FPS | 583.7 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9982 | 0.9959 | 0.9959 | 0.9982 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9594 | 0.9981 | 0.9962 | 0.9774 | 0.9790 | 0.9986 | 0.9972 | 271 |
| PNEUMONIA | 0.9926 | 0.9767 | 0.9571 | 0.9745 | 0.9960 | 0.9983 | 0.9972 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=540 FN=1
- **NORMAL**: TP=260 FP=1 TN=513 FN=11
- **PNEUMONIA**: TP=268 FP=12 TN=503 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 260 | 11 |
| PNEUMONIA | 1 | 1 | 268 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9962    0.9594    0.9774       271
   PNEUMONIA     0.9571    0.9926    0.9745       270

    accuracy                         0.9822       785
   macro avg     0.9831    0.9826    0.9826       785
weighted avg     0.9827    0.9822    0.9822       785

```