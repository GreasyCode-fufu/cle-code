# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/repvit_m1_seed44/repvit_m1/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9724 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9964 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9719 |
| Balanced Acc | 0.9724 |
| Macro-Sensitivity | 0.9724 |
| Macro-Specificity | 0.9858 |
| Macro-Precision | 0.9724 |
| AUC-PR (Macro) | 0.9931 |
| Cohen's Kappa | 0.9579 |
| MCC | 0.9579 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3882.9 (0.26 ms/张) |
| End-to-End FPS | 739.9 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9877 | 0.9944 | 0.9877 | 0.9877 | 0.9944 | 0.9989 | 0.9980 | 244 |
| NORMAL | 0.9704 | 0.9825 | 0.9668 | 0.9686 | 0.9844 | 0.9951 | 0.9943 | 270 |
| PNEUMONIA | 0.9593 | 0.9805 | 0.9628 | 0.9610 | 0.9786 | 0.9951 | 0.9871 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=241 FP=3 TN=537 FN=3
- **NORMAL**: TP=262 FP=9 TN=505 FN=8
- **PNEUMONIA**: TP=259 FP=10 TN=504 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 241 | 1 | 2 |
| NORMAL | 0 | 262 | 8 |
| PNEUMONIA | 3 | 8 | 259 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9877    0.9877    0.9877       244
      NORMAL     0.9668    0.9704    0.9686       270
   PNEUMONIA     0.9628    0.9593    0.9610       270

    accuracy                         0.9719       784
   macro avg     0.9724    0.9724    0.9724       784
weighted avg     0.9719    0.9719    0.9719       784

```