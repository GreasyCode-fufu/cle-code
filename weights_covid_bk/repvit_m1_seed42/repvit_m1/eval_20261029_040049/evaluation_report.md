# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/repvit_m1_seed42/repvit_m1/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9714 |
| Accuracy | 0.9707 |
| Macro-AUROC (OvR) | 0.9976 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9707 |
| Balanced Acc | 0.9714 |
| Macro-Sensitivity | 0.9714 |
| Macro-Specificity | 0.9851 |
| Macro-Precision | 0.9718 |
| AUC-PR (Macro) | 0.9957 |
| Cohen's Kappa | 0.9560 |
| MCC | 0.9561 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3919.0 (0.26 ms/张) |
| End-to-End FPS | 755.6 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9982 | 0.9959 | 0.9938 | 0.9963 | 0.9999 | 0.9999 | 244 |
| NORMAL | 0.9779 | 0.9728 | 0.9498 | 0.9636 | 0.9881 | 0.9967 | 0.9941 | 271 |
| PNEUMONIA | 0.9444 | 0.9845 | 0.9696 | 0.9568 | 0.9713 | 0.9963 | 0.9931 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=1 TN=540 FN=2
- **NORMAL**: TP=265 FP=14 TN=500 FN=6
- **PNEUMONIA**: TP=255 FP=8 TN=507 FN=15

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 0 | 2 |
| NORMAL | 0 | 265 | 6 |
| PNEUMONIA | 1 | 14 | 255 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9918    0.9938       244
      NORMAL     0.9498    0.9779    0.9636       271
   PNEUMONIA     0.9696    0.9444    0.9568       270

    accuracy                         0.9707       785
   macro avg     0.9718    0.9714    0.9714       785
weighted avg     0.9709    0.9707    0.9707       785

```