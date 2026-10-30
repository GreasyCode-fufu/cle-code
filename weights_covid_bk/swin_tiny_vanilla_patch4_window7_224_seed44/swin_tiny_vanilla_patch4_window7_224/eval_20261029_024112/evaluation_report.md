# swin_tiny_vanilla_patch4_window7_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/swin_tiny_vanilla_patch4_window7_224_seed44/swin_tiny_vanilla_patch4_window7_224/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9813 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9970 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9809 |
| Balanced Acc | 0.9813 |
| Macro-Sensitivity | 0.9813 |
| Macro-Specificity | 0.9903 |
| Macro-Precision | 0.9817 |
| AUC-PR (Macro) | 0.9955 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9715 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.37G |
| FLOPs | 8.74G |
| Params | 27.50M |
| Pure-Inference FPS | 1554.4 (0.64 ms/张) |
| End-to-End FPS | 586.5 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9963 | 0.9918 | 0.9939 | 0.9981 | 0.9967 | 0.9968 | 244 |
| NORMAL | 0.9593 | 0.9981 | 0.9962 | 0.9774 | 0.9790 | 0.9982 | 0.9971 | 270 |
| PNEUMONIA | 0.9889 | 0.9767 | 0.9570 | 0.9727 | 0.9941 | 0.9960 | 0.9925 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=2 TN=538 FN=1
- **NORMAL**: TP=259 FP=1 TN=513 FN=11
- **PNEUMONIA**: TP=267 FP=12 TN=502 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 259 | 11 |
| PNEUMONIA | 2 | 1 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9918    0.9959    0.9939       244
      NORMAL     0.9962    0.9593    0.9774       270
   PNEUMONIA     0.9570    0.9889    0.9727       270

    accuracy                         0.9809       784
   macro avg     0.9817    0.9813    0.9813       784
weighted avg     0.9813    0.9809    0.9809       784

```