# mbv3_lp_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6/mbv3_lp_dce4m_seed42/mbv3_lp_dce4m/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9762 |
| Accuracy | 0.9758 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9757 |
| Balanced Acc | 0.9763 |
| Macro-Sensitivity | 0.9763 |
| Macro-Specificity | 0.9878 |
| Macro-Precision | 0.9762 |
| AUC-PR (Macro) | 0.9921 |
| Cohen's Kappa | 0.9636 |
| MCC | 0.9637 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 2488.8 (0.40 ms/张) |
| End-to-End FPS | 16.4 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9944 | 0.9878 | 0.9898 | 0.9963 | 0.9968 | 0.9971 | 244 |
| NORMAL | 0.9852 | 0.9805 | 0.9638 | 0.9744 | 0.9921 | 0.9969 | 0.9955 | 270 |
| PNEUMONIA | 0.9519 | 0.9883 | 0.9772 | 0.9644 | 0.9750 | 0.9942 | 0.9837 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=3 TN=537 FN=2
- **NORMAL**: TP=266 FP=10 TN=504 FN=4
- **PNEUMONIA**: TP=257 FP=6 TN=508 FN=13

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 0 | 2 |
| NORMAL | 0 | 266 | 4 |
| PNEUMONIA | 3 | 10 | 257 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9878    0.9918    0.9898       244
      NORMAL     0.9638    0.9852    0.9744       270
   PNEUMONIA     0.9772    0.9519    0.9644       270

    accuracy                         0.9758       784
   macro avg     0.9762    0.9763    0.9762       784
weighted avg     0.9759    0.9758    0.9757       784

```