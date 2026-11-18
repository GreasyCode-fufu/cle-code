# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/convnext_small_seed43/convnext_small/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9851 |
| Accuracy | 0.9847 |
| Macro-AUROC (OvR) | 0.9982 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9847 |
| Balanced Acc | 0.9851 |
| Macro-Sensitivity | 0.9851 |
| Macro-Specificity | 0.9922 |
| Macro-Precision | 0.9851 |
| AUC-PR (Macro) | 0.9968 |
| Cohen's Kappa | 0.9770 |
| MCC | 0.9771 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 1059.1 (0.94 ms/张) |
| End-to-End FPS | 498.9 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9986 | 0.9982 | 244 |
| NORMAL | 0.9741 | 0.9942 | 0.9887 | 0.9813 | 0.9865 | 0.9986 | 0.9977 | 270 |
| PNEUMONIA | 0.9852 | 0.9844 | 0.9708 | 0.9779 | 0.9922 | 0.9975 | 0.9944 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=263 FP=3 TN=511 FN=7
- **PNEUMONIA**: TP=266 FP=8 TN=506 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 263 | 7 |
| PNEUMONIA | 1 | 3 | 266 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9887    0.9741    0.9813       270
   PNEUMONIA     0.9708    0.9852    0.9779       270

    accuracy                         0.9847       784
   macro avg     0.9851    0.9851    0.9851       784
weighted avg     0.9848    0.9847    0.9847       784

```