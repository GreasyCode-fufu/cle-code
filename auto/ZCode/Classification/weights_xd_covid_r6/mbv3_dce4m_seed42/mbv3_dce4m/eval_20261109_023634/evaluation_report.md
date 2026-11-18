# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6/mbv3_dce4m_seed42/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9875 |
| Accuracy | 0.9872 |
| Macro-AUROC (OvR) | 0.9981 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9872 |
| Balanced Acc | 0.9875 |
| Macro-Sensitivity | 0.9875 |
| Macro-Specificity | 0.9935 |
| Macro-Precision | 0.9875 |
| AUC-PR (Macro) | 0.9965 |
| Cohen's Kappa | 0.9808 |
| MCC | 0.9809 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 3295.5 (0.30 ms/张) |
| End-to-End FPS | 15.2 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9981 | 0.9977 | 244 |
| NORMAL | 0.9889 | 0.9903 | 0.9816 | 0.9852 | 0.9941 | 0.9988 | 0.9978 | 270 |
| PNEUMONIA | 0.9778 | 0.9922 | 0.9851 | 0.9814 | 0.9884 | 0.9973 | 0.9939 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=267 FP=5 TN=509 FN=3
- **PNEUMONIA**: TP=264 FP=4 TN=510 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 267 | 3 |
| PNEUMONIA | 1 | 5 | 264 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9816    0.9889    0.9852       270
   PNEUMONIA     0.9851    0.9778    0.9814       270

    accuracy                         0.9872       784
   macro avg     0.9875    0.9875    0.9875       784
weighted avg     0.9873    0.9872    0.9872       784

```