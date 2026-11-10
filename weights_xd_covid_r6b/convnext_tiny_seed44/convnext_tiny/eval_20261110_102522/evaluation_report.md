# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/convnext_tiny_seed44/convnext_tiny/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9876 |
| Accuracy | 0.9873 |
| Macro-AUROC (OvR) | 0.9996 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9873 |
| Balanced Acc | 0.9874 |
| Macro-Sensitivity | 0.9874 |
| Macro-Specificity | 0.9935 |
| Macro-Precision | 0.9879 |
| AUC-PR (Macro) | 0.9993 |
| Cohen's Kappa | 0.9809 |
| MCC | 0.9810 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 594.0 (1.68 ms/张) |
| End-to-End FPS | 363.8 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 1.0000 | 1.0000 | 244 |
| NORMAL | 0.9742 | 0.9961 | 0.9925 | 0.9832 | 0.9865 | 0.9994 | 0.9988 | 271 |
| PNEUMONIA | 0.9963 | 0.9845 | 0.9711 | 0.9835 | 0.9980 | 0.9995 | 0.9990 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=264 FP=2 TN=512 FN=7
- **PNEUMONIA**: TP=269 FP=8 TN=507 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 264 | 7 |
| PNEUMONIA | 0 | 1 | 269 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9925    0.9742    0.9832       271
   PNEUMONIA     0.9711    0.9963    0.9835       270

    accuracy                         0.9873       785
   macro avg     0.9879    0.9874    0.9876       785
weighted avg     0.9875    0.9873    0.9873       785

```