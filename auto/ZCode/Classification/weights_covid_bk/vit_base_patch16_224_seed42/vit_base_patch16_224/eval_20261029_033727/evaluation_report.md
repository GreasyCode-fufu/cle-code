# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/vit_base_patch16_224_seed42/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9776 |
| Accuracy | 0.9770 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9770 |
| Balanced Acc | 0.9776 |
| Macro-Sensitivity | 0.9776 |
| Macro-Specificity | 0.9884 |
| Macro-Precision | 0.9777 |
| AUC-PR (Macro) | 0.9941 |
| Cohen's Kappa | 0.9655 |
| MCC | 0.9655 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 845.5 (1.18 ms/张) |
| End-to-End FPS | 448.1 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9969 | 0.9970 | 244 |
| NORMAL | 0.9741 | 0.9825 | 0.9669 | 0.9705 | 0.9863 | 0.9970 | 0.9949 | 270 |
| PNEUMONIA | 0.9630 | 0.9844 | 0.9701 | 0.9665 | 0.9806 | 0.9950 | 0.9902 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=263 FP=9 TN=505 FN=7
- **PNEUMONIA**: TP=260 FP=8 TN=506 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 263 | 7 |
| PNEUMONIA | 1 | 9 | 260 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9669    0.9741    0.9705       270
   PNEUMONIA     0.9701    0.9630    0.9665       270

    accuracy                         0.9770       784
   macro avg     0.9777    0.9776    0.9776       784
weighted avg     0.9770    0.9770    0.9770       784

```