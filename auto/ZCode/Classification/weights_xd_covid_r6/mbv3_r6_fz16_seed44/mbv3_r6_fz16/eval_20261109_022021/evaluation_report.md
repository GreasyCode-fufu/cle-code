# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6/mbv3_r6_fz16_seed44/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9851 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9973 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9851 |
| Macro-Sensitivity | 0.9851 |
| Macro-Specificity | 0.9922 |
| Macro-Precision | 0.9851 |
| AUC-PR (Macro) | 0.9956 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9770 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4146.9 (0.24 ms/张) |
| End-to-End FPS | 662.7 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9961 | 0.9972 | 244 |
| NORMAL | 0.9778 | 0.9922 | 0.9851 | 0.9814 | 0.9884 | 0.9985 | 0.9976 | 270 |
| PNEUMONIA | 0.9815 | 0.9864 | 0.9743 | 0.9779 | 0.9902 | 0.9972 | 0.9922 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=264 FP=4 TN=510 FN=6
- **PNEUMONIA**: TP=265 FP=7 TN=507 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 264 | 6 |
| PNEUMONIA | 1 | 4 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9851    0.9778    0.9814       270
   PNEUMONIA     0.9743    0.9815    0.9779       270

    accuracy                         0.9847       784
   macro avg     0.9851    0.9851    0.9851       784
weighted avg     0.9847    0.9847    0.9847       784

```