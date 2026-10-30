# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/vit_base_patch16_224_seed43/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9812 |
| Accuracy | 0.9809 |
| Macro-AUROC (OvR) | 0.9933 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9808 |
| Balanced Acc | 0.9813 |
| Macro-Sensitivity | 0.9813 |
| Macro-Specificity | 0.9904 |
| Macro-Precision | 0.9811 |
| AUC-PR (Macro) | 0.9885 |
| Cohen's Kappa | 0.9713 |
| MCC | 0.9713 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 848.4 (1.18 ms/张) |
| End-to-End FPS | 443.2 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9944 | 0.9878 | 0.9918 | 0.9981 | 0.9932 | 0.9866 | 244 |
| NORMAL | 0.9815 | 0.9883 | 0.9779 | 0.9797 | 0.9903 | 0.9976 | 0.9961 | 270 |
| PNEUMONIA | 0.9667 | 0.9883 | 0.9775 | 0.9721 | 0.9826 | 0.9890 | 0.9829 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=3 TN=537 FN=1
- **NORMAL**: TP=265 FP=6 TN=508 FN=5
- **PNEUMONIA**: TP=261 FP=6 TN=508 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 265 | 5 |
| PNEUMONIA | 3 | 6 | 261 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9878    0.9959    0.9918       244
      NORMAL     0.9779    0.9815    0.9797       270
   PNEUMONIA     0.9775    0.9667    0.9721       270

    accuracy                         0.9809       784
   macro avg     0.9811    0.9813    0.9812       784
weighted avg     0.9808    0.9809    0.9808       784

```