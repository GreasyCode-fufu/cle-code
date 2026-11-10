# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/gmnet_s3_seed44/gmnet_s3/model_best.pth`
- 测试集: `./datasets/covid_v1/val`  (共 784 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9801 |
| Accuracy | 0.9796 |
| Macro-AUROC (OvR) | 0.9967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9796 |
| Balanced Acc | 0.9801 |
| Macro-Sensitivity | 0.9801 |
| Macro-Specificity | 0.9897 |
| Macro-Precision | 0.9801 |
| AUC-PR (Macro) | 0.9943 |
| Cohen's Kappa | 0.9694 |
| MCC | 0.9694 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1239.9 (0.81 ms/张) |
| End-to-End FPS | 525.4 |
| 测试样本数 | 784 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 0.9981 | 0.9959 | 0.9959 | 0.9981 | 0.9976 | 0.9977 | 244 |
| NORMAL | 0.9741 | 0.9864 | 0.9741 | 0.9741 | 0.9864 | 0.9971 | 0.9939 | 270 |
| PNEUMONIA | 0.9704 | 0.9844 | 0.9704 | 0.9704 | 0.9844 | 0.9955 | 0.9914 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=1 TN=539 FN=1
- **NORMAL**: TP=263 FP=7 TN=507 FN=7
- **PNEUMONIA**: TP=262 FP=8 TN=506 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 0 | 1 |
| NORMAL | 0 | 263 | 7 |
| PNEUMONIA | 1 | 7 | 262 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     0.9959    0.9959    0.9959       244
      NORMAL     0.9741    0.9741    0.9741       270
   PNEUMONIA     0.9704    0.9704    0.9704       270

    accuracy                         0.9796       784
   macro avg     0.9801    0.9801    0.9801       784
weighted avg     0.9796    0.9796    0.9796       784

```