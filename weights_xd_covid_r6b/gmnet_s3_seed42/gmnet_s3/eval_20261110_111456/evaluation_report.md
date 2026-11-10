# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/gmnet_s3_seed42/gmnet_s3/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9800 |
| Accuracy | 0.9796 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9796 |
| Balanced Acc | 0.9800 |
| Macro-Sensitivity | 0.9800 |
| Macro-Specificity | 0.9897 |
| Macro-Precision | 0.9800 |
| AUC-PR (Macro) | 0.9922 |
| Cohen's Kappa | 0.9694 |
| MCC | 0.9694 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1182.6 (0.85 ms/张) |
| End-to-End FPS | 512.4 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9963 | 0.9918 | 0.9918 | 0.9963 | 0.9961 | 0.9972 | 244 |
| NORMAL | 0.9778 | 0.9864 | 0.9742 | 0.9760 | 0.9883 | 0.9974 | 0.9950 | 270 |
| PNEUMONIA | 0.9704 | 0.9864 | 0.9740 | 0.9722 | 0.9845 | 0.9954 | 0.9845 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=2 TN=538 FN=2
- **NORMAL**: TP=264 FP=7 TN=507 FN=6
- **PNEUMONIA**: TP=262 FP=7 TN=507 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 264 | 6 |
| PNEUMONIA | 2 | 6 | 262 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9918    0.9918    0.9918       244
      NORMAL     0.9742    0.9778    0.9760       270
   PNEUMONIA     0.9740    0.9704    0.9722       270

    accuracy                         0.9796       784
   macro avg     0.9800    0.9800    0.9800       784
weighted avg     0.9796    0.9796    0.9796       784

```