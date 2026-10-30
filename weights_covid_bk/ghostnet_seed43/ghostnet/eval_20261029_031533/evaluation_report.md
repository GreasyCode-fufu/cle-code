# ghostnet 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/ghostnet_seed43/ghostnet/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9827 |
| Accuracy | 0.9822 |
| Macro-AUROC (OvR) | 0.9991 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9822 |
| Balanced Acc | 0.9826 |
| Macro-Sensitivity | 0.9826 |
| Macro-Specificity | 0.9909 |
| Macro-Precision | 0.9828 |
| AUC-PR (Macro) | 0.9983 |
| Cohen's Kappa | 0.9732 |
| MCC | 0.9732 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 154.58M |
| FLOPs | 309.16M |
| Params | 3.91M |
| Pure-Inference FPS | 5039.8 (0.20 ms/张) |
| End-to-End FPS | 792.8 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 1.0000 | 1.0000 | 0.9979 | 0.9982 | 0.9999 | 0.9999 | 244 |
| NORMAL | 0.9742 | 0.9883 | 0.9778 | 0.9760 | 0.9864 | 0.9988 | 0.9979 | 271 |
| PNEUMONIA | 0.9778 | 0.9845 | 0.9706 | 0.9742 | 0.9883 | 0.9985 | 0.9972 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=0 TN=541 FN=1
- **NORMAL**: TP=264 FP=6 TN=508 FN=7
- **PNEUMONIA**: TP=264 FP=8 TN=507 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 264 | 7 |
| PNEUMONIA | 0 | 6 | 264 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9959    0.9979       244
      NORMAL     0.9778    0.9742    0.9760       271
   PNEUMONIA     0.9706    0.9778    0.9742       270

    accuracy                         0.9822       785
   macro avg     0.9828    0.9826    0.9827       785
weighted avg     0.9822    0.9822    0.9822       785

```