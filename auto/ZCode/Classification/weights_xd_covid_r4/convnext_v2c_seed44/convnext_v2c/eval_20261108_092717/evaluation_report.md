# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r4/convnext_v2c_seed44/convnext_v2c/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9913 |
| Accuracy | 0.9911 |
| Macro-AUROC (OvR) | 0.9995 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9911 |
| Balanced Acc | 0.9912 |
| Macro-Sensitivity | 0.9912 |
| Macro-Specificity | 0.9955 |
| Macro-Precision | 0.9914 |
| AUC-PR (Macro) | 0.9990 |
| Cohen's Kappa | 0.9866 |
| MCC | 0.9866 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 696.8 (1.44 ms/张) |
| End-to-End FPS | 394.7 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 1.0000 | 1.0000 | 0.9979 | 0.9982 | 0.9998 | 0.9997 | 244 |
| NORMAL | 0.9852 | 0.9961 | 0.9926 | 0.9889 | 0.9922 | 0.9994 | 0.9989 | 271 |
| PNEUMONIA | 0.9926 | 0.9903 | 0.9817 | 0.9871 | 0.9961 | 0.9991 | 0.9984 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=0 TN=541 FN=1
- **NORMAL**: TP=267 FP=2 TN=512 FN=4
- **PNEUMONIA**: TP=268 FP=5 TN=510 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 267 | 4 |
| PNEUMONIA | 0 | 2 | 268 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9959    0.9979       244
      NORMAL     0.9926    0.9852    0.9889       271
   PNEUMONIA     0.9817    0.9926    0.9871       270

    accuracy                         0.9911       785
   macro avg     0.9914    0.9912    0.9913       785
weighted avg     0.9911    0.9911    0.9911       785

```