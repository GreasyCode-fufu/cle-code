# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/resnet50_seed43/resnet50/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9838 |
| Accuracy | 0.9834 |
| Macro-AUROC (OvR) | 0.9973 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9834 |
| Balanced Acc | 0.9836 |
| Macro-Sensitivity | 0.9836 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9841 |
| AUC-PR (Macro) | 0.9939 |
| Cohen's Kappa | 0.9751 |
| MCC | 0.9751 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 2163.7 (0.46 ms/张) |
| End-to-End FPS | 654.3 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9877 | 1.0000 | 1.0000 | 0.9938 | 0.9945 | 0.9975 | 0.9976 | 244 |
| NORMAL | 0.9852 | 0.9844 | 0.9708 | 0.9779 | 0.9922 | 0.9979 | 0.9961 | 270 |
| PNEUMONIA | 0.9778 | 0.9903 | 0.9814 | 0.9796 | 0.9883 | 0.9965 | 0.9880 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=241 FP=0 TN=540 FN=3
- **NORMAL**: TP=266 FP=8 TN=506 FN=4
- **PNEUMONIA**: TP=264 FP=5 TN=509 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 241 | 2 | 1 |
| NORMAL | 0 | 266 | 4 |
| PNEUMONIA | 0 | 6 | 264 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9877    0.9938       244
      NORMAL     0.9708    0.9852    0.9779       270
   PNEUMONIA     0.9814    0.9778    0.9796       270

    accuracy                         0.9834       784
   macro avg     0.9841    0.9836    0.9838       784
weighted avg     0.9835    0.9834    0.9834       784

```