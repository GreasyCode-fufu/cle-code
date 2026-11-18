# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/iformer_s_seed44/iformer_s/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9875 |
| Accuracy | 0.9872 |
| Macro-AUROC (OvR) | 0.9971 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9872 |
| Balanced Acc | 0.9875 |
| Macro-Sensitivity | 0.9875 |
| Macro-Specificity | 0.9935 |
| Macro-Precision | 0.9875 |
| AUC-PR (Macro) | 0.9944 |
| Cohen's Kappa | 0.9808 |
| MCC | 0.9808 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 2825.3 (0.35 ms/张) |
| End-to-End FPS | 672.7 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9963 | 0.9969 | 244 |
| NORMAL | 0.9852 | 0.9922 | 0.9852 | 0.9852 | 0.9922 | 0.9986 | 0.9977 | 270 |
| PNEUMONIA | 0.9815 | 0.9903 | 0.9815 | 0.9815 | 0.9903 | 0.9964 | 0.9887 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=266 FP=4 TN=510 FN=4
- **PNEUMONIA**: TP=265 FP=5 TN=509 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 266 | 4 |
| PNEUMONIA | 1 | 4 | 265 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9852    0.9852    0.9852       270
   PNEUMONIA     0.9815    0.9815    0.9815       270

    accuracy                         0.9872       784
   macro avg     0.9875    0.9875    0.9875       784
weighted avg     0.9872    0.9872    0.9872       784

```