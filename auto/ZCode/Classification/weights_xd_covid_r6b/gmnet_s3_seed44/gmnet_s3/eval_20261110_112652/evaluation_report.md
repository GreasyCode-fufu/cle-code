# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_covid_r6b/gmnet_s3_seed44/gmnet_s3/model_best.pth`
- 测试集: `./datasets/covid_v1/test`  (共 785 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9839 |
| Accuracy | 0.9834 |
| Macro-AUROC (OvR) | 0.9979 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9834 |
| Balanced Acc | 0.9839 |
| Macro-Sensitivity | 0.9839 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9840 |
| AUC-PR (Macro) | 0.9959 |
| Cohen's Kappa | 0.9751 |
| MCC | 0.9752 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1421.8 (0.70 ms/张) |
| End-to-End FPS | 549.2 |
| 测试样本数 | 785 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| COVID | 0.9959 | 1.0000 | 1.0000 | 0.9979 | 0.9982 | 0.9987 | 0.9983 | 244 |
| NORMAL | 0.9705 | 0.9903 | 0.9813 | 0.9759 | 0.9845 | 0.9969 | 0.9927 | 271 |
| PNEUMONIA | 0.9852 | 0.9845 | 0.9708 | 0.9779 | 0.9922 | 0.9979 | 0.9967 | 270 |

### 每类 TP/FP/TN/FN

- **COVID**: TP=243 FP=0 TN=541 FN=1
- **NORMAL**: TP=263 FP=5 TN=509 FN=8
- **PNEUMONIA**: TP=266 FP=8 TN=507 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | COVID | NORMAL | PNEUMONIA |
|---|---|---|---|
| COVID | 243 | 1 | 0 |
| NORMAL | 0 | 263 | 8 |
| PNEUMONIA | 0 | 4 | 266 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

       COVID     1.0000    0.9959    0.9979       244
      NORMAL     0.9813    0.9705    0.9759       271
   PNEUMONIA     0.9708    0.9852    0.9779       270

    accuracy                         0.9834       785
   macro avg     0.9840    0.9839    0.9839       785
weighted avg     0.9835    0.9834    0.9834       785

```