# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/iformer_s_seed44/iformer_s/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9826 |
| Accuracy | 0.9822 |
| Macro-AUROC (OvR) | 0.9984 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9822 |
| Balanced Acc | 0.9825 |
| Macro-Sensitivity | 0.9825 |
| Macro-Specificity | 0.9909 |
| Macro-Precision | 0.9828 |
| AUC-PR (Macro) | 0.9972 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 2932.5 (0.34 ms/张) |
| End-to-End FPS | 671.9 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9742 | 0.9883 | 0.9778 | 0.9760 | 0.9864 | 0.9975 | 0.9966 | 271 |
| PNEUMONIA | 0.9815 | 0.9845 | 0.9707 | 0.9761 | 0.9902 | 0.9978 | 0.9950 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=264 FP=6 TN=508 FN=7
- **PNEUMONIA**: TP=265 FP=8 TN=507 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 264 | 7 |
| PNEUMONIA | 0 | 5 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9778    0.9742    0.9760       271
   PNEUMONIA     0.9707    0.9815    0.9761       270

    accuracy                         0.9822       785
   macro avg     0.9828    0.9825    0.9826       785
weighted avg     0.9822    0.9822    0.9822       785

```