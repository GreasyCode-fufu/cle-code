# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/resnet34_seed44/resnet34/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9801 |
| Accuracy | 0.9796 |
| Macro-AUROC (OvR) | 0.9992 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9796 |
| Balanced Acc | 0.9800 |
| Macro-Sensitivity | 0.9800 |
| Macro-Specificity | 0.9897 |
| Macro-Precision | 0.9804 |
| AUC-PR (Macro) | 0.9985 |
| Cohen's Kappa | 0.9694 |
| MCC | 0.9695 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 3842.8 (0.26 ms/张) |
| End-to-End FPS | 754.0 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 0.9982 | 0.9959 | 0.9938 | 0.9963 | 1.0000 | 0.9999 | 244 |
| NORMAL | 0.9631 | 0.9922 | 0.9849 | 0.9739 | 0.9808 | 0.9990 | 0.9981 | 271 |
| PNEUMONIA | 0.9852 | 0.9786 | 0.9603 | 0.9726 | 0.9921 | 0.9985 | 0.9974 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=1 TN=540 FN=2
- **NORMAL**: TP=261 FP=4 TN=510 FN=10
- **PNEUMONIA**: TP=266 FP=11 TN=504 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 261 | 10 |
| PNEUMONIA | 1 | 3 | 266 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9918    0.9938       244
      NORMAL     0.9849    0.9631    0.9739       271
   PNEUMONIA     0.9603    0.9852    0.9726       270

    accuracy                         0.9796       785
   macro avg     0.9804    0.9800    0.9801       785
weighted avg     0.9799    0.9796    0.9796       785

```