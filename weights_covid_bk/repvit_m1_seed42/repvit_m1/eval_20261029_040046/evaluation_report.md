# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/repvit_m1_seed42/repvit_m1/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9700 |
| Accuracy | 0.9694 |
| Macro-AUROC (OvR) | 0.9954 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9694 |
| Balanced Acc | 0.9700 |
| Macro-Sensitivity | 0.9700 |
| Macro-Specificity | 0.9845 |
| Macro-Precision | 0.9702 |
| AUC-PR (Macro) | 0.9916 |
| Cohen's Kappa | 0.9540 |
| MCC | 0.9542 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 3935.6 (0.25 ms/张) |
| End-to-End FPS | 756.2 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9877 | 0.9944 | 0.9877 | 0.9877 | 0.9944 | 0.9991 | 0.9979 | 244 |
| NORMAL | 0.9778 | 0.9728 | 0.9496 | 0.9635 | 0.9881 | 0.9945 | 0.9907 | 270 |
| PNEUMONIA | 0.9444 | 0.9864 | 0.9733 | 0.9586 | 0.9713 | 0.9926 | 0.9863 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=241 FP=3 TN=537 FN=3
- **NORMAL**: TP=264 FP=14 TN=500 FN=6
- **PNEUMONIA**: TP=255 FP=7 TN=507 FN=15

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 241 | 2 | 1 |
| NORMAL | 0 | 264 | 6 |
| PNEUMONIA | 3 | 12 | 255 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9877    0.9877    0.9877       244
      NORMAL     0.9496    0.9778    0.9635       270
   PNEUMONIA     0.9733    0.9444    0.9586       270

    accuracy                         0.9694       784
   macro avg     0.9702    0.9700    0.9700       784
weighted avg     0.9696    0.9694    0.9694       784

```