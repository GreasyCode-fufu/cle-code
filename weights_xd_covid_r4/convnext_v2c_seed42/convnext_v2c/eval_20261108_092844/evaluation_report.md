# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r4/convnext_v2c_seed42/convnext_v2c/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9825 |
| Accuracy | 0.9821 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9821 |
| Balanced Acc | 0.9825 |
| Macro-Sensitivity | 0.9825 |
| Macro-Specificity | 0.9910 |
| Macro-Precision | 0.9826 |
| AUC-PR (Macro) | 0.9937 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 841.0 (1.19 ms/张) |
| End-to-End FPS | 439.1 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9981 | 0.9959 | 0.9938 | 0.9963 | 0.9962 | 0.9971 | 244 |
| NORMAL | 0.9852 | 0.9864 | 0.9744 | 0.9797 | 0.9922 | 0.9976 | 0.9960 | 270 |
| PNEUMONIA | 0.9704 | 0.9883 | 0.9776 | 0.9740 | 0.9845 | 0.9959 | 0.9879 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=1 TN=539 FN=2
- **NORMAL**: TP=266 FP=7 TN=507 FN=4
- **PNEUMONIA**: TP=262 FP=6 TN=508 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 0 | 2 |
| NORMAL | 0 | 266 | 4 |
| PNEUMONIA | 1 | 7 | 262 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9918    0.9938       244
      NORMAL     0.9744    0.9852    0.9797       270
   PNEUMONIA     0.9776    0.9704    0.9740       270

    accuracy                         0.9821       784
   macro avg     0.9826    0.9825    0.9825       784
weighted avg     0.9822    0.9821    0.9821       784

```