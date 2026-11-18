# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_covid_bk/vit_base_patch16_224_seed44/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9763 |
| Accuracy | 0.9758 |
| Macro-AUROC (OvR) | 0.9988 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9758 |
| Balanced Acc | 0.9762 |
| Macro-Sensitivity | 0.9762 |
| Macro-Specificity | 0.9877 |
| Macro-Precision | 0.9765 |
| AUC-PR (Macro) | 0.9978 |
| Cohen's Kappa | 0.9636 |
| MCC | 0.9637 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 848.2 (1.18 ms/张) |
| End-to-End FPS | 443.1 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9877 | 0.9982 | 0.9959 | 0.9918 | 0.9945 | 0.9996 | 0.9993 | 244 |
| NORMAL | 0.9742 | 0.9805 | 0.9635 | 0.9688 | 0.9863 | 0.9985 | 0.9972 | 271 |
| PNEUMONIA | 0.9667 | 0.9845 | 0.9703 | 0.9685 | 0.9826 | 0.9984 | 0.9970 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=241 FP=1 TN=540 FN=3
- **NORMAL**: TP=264 FP=10 TN=504 FN=7
- **PNEUMONIA**: TP=261 FP=8 TN=507 FN=9

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 241 | 2 | 1 |
| NORMAL | 0 | 264 | 7 |
| PNEUMONIA | 1 | 8 | 261 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9877    0.9918       244
      NORMAL     0.9635    0.9742    0.9688       271
   PNEUMONIA     0.9703    0.9667    0.9685       270

    accuracy                         0.9758       785
   macro avg     0.9765    0.9762    0.9763       785
weighted avg     0.9759    0.9758    0.9758       785

```