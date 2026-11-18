# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/gmnet_s3_seed43/gmnet_s3/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9888 |
| Accuracy | 0.9885 |
| Macro-AUROC (OvR) | 0.9986 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9885 |
| Balanced Acc | 0.9886 |
| Macro-Sensitivity | 0.9886 |
| Macro-Specificity | 0.9942 |
| Macro-Precision | 0.9890 |
| AUC-PR (Macro) | 0.9974 |
| Cohen's Kappa | 0.9828 |
| MCC | 0.9828 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1644.2 (0.61 ms/张) |
| End-to-End FPS | 582.8 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9918 | 1.0000 | 1.0000 | 0.9959 | 0.9963 | 0.9999 | 0.9999 | 244 |
| NORMAL | 0.9852 | 0.9922 | 0.9852 | 0.9852 | 0.9922 | 0.9981 | 0.9955 | 271 |
| PNEUMONIA | 0.9889 | 0.9903 | 0.9816 | 0.9852 | 0.9942 | 0.9977 | 0.9968 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=242 FP=0 TN=541 FN=2
- **NORMAL**: TP=267 FP=4 TN=510 FN=4
- **PNEUMONIA**: TP=267 FP=5 TN=510 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 242 | 1 | 1 |
| NORMAL | 0 | 267 | 4 |
| PNEUMONIA | 0 | 3 | 267 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9918    0.9959       244
      NORMAL     0.9852    0.9852    0.9852       271
   PNEUMONIA     0.9816    0.9889    0.9852       270

    accuracy                         0.9885       785
   macro avg     0.9890    0.9886    0.9888       785
weighted avg     0.9886    0.9885    0.9885       785

```