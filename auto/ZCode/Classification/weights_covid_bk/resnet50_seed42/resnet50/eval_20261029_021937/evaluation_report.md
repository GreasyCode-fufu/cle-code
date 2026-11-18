# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/resnet50_seed42/resnet50/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9875 |
| Accuracy | 0.9872 |
| Macro-AUROC (OvR) | 0.9972 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9873 |
| Balanced Acc | 0.9875 |
| Macro-Sensitivity | 0.9875 |
| Macro-Specificity | 0.9935 |
| Macro-Precision | 0.9877 |
| AUC-PR (Macro) | 0.9955 |
| Cohen's Kappa | 0.9808 |
| MCC | 0.9809 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 2148.2 (0.47 ms/张) |
| End-to-End FPS | 656.3 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9961 | 0.9972 | 244 |
| NORMAL | 0.9741 | 0.9981 | 0.9962 | 0.9850 | 0.9865 | 0.9983 | 0.9973 | 270 |
| PNEUMONIA | 0.9926 | 0.9844 | 0.9710 | 0.9817 | 0.9961 | 0.9972 | 0.9921 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=263 FP=1 TN=513 FN=7
- **PNEUMONIA**: TP=268 FP=8 TN=506 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 263 | 7 |
| PNEUMONIA | 1 | 1 | 268 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9962    0.9741    0.9850       270
   PNEUMONIA     0.9710    0.9926    0.9817       270

    accuracy                         0.9872       784
   macro avg     0.9877    0.9875    0.9875       784
weighted avg     0.9874    0.9872    0.9873       784

```