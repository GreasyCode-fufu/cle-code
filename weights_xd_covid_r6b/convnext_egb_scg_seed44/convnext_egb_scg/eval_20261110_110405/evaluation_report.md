# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/convnext_egb_scg_seed44/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9864 |
| Accuracy | 0.9860 |
| Macro-AUROC (OvR) | 0.9997 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9860 |
| Balanced Acc | 0.9863 |
| Macro-Sensitivity | 0.9863 |
| Macro-Specificity | 0.9929 |
| Macro-Precision | 0.9867 |
| AUC-PR (Macro) | 0.9994 |
| Cohen's Kappa | 0.9790 |
| MCC | 0.9791 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 359.1 (2.78 ms/张) |
| End-to-End FPS | 258.8 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 1.0000 | 1.0000 | 0.9979 | 0.9982 | 1.0000 | 1.0000 | 244 |
| NORMAL | 0.9705 | 0.9961 | 0.9925 | 0.9813 | 0.9846 | 0.9995 | 0.9990 | 271 |
| PNEUMONIA | 0.9926 | 0.9825 | 0.9675 | 0.9799 | 0.9961 | 0.9995 | 0.9990 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=0 TN=541 FN=1
- **NORMAL**: TP=263 FP=2 TN=512 FN=8
- **PNEUMONIA**: TP=268 FP=9 TN=506 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 263 | 8 |
| PNEUMONIA | 0 | 2 | 268 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9959    0.9979       244
      NORMAL     0.9925    0.9705    0.9813       271
   PNEUMONIA     0.9675    0.9926    0.9799       270

    accuracy                         0.9860       785
   macro avg     0.9867    0.9863    0.9864       785
weighted avg     0.9862    0.9860    0.9860       785

```