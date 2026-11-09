# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/iformer_s_seed46/iformer_s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9412 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9938 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9411 |
| Balanced Acc | 0.9406 |
| Macro-Sensitivity | 0.9406 |
| Macro-Specificity | 0.9702 |
| Macro-Precision | 0.9420 |
| AUC-PR (Macro) | 0.9887 |
| Cohen's Kappa | 0.9112 |
| MCC | 0.9113 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 3136.9 (0.32 ms/张) |
| End-to-End FPS | 221.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9484 | 0.9108 | 0.9216 | 0.9614 | 0.9908 | 0.9849 | 208 |
| Malignant_lesions | 0.9205 | 0.9700 | 0.9310 | 0.9257 | 0.9652 | 0.9916 | 0.9833 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9922 | 0.9841 | 0.9764 | 0.9845 | 0.9990 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=19 TN=349 FN=14
- **Malignant_lesions**: TP=162 FP=12 TN=388 FN=14
- **body_of_stomach_normal**: TP=186 FP=3 TN=381 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 11 | 3 |
| Malignant_lesions | 14 | 162 | 0 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9108    0.9327    0.9216       208
     Malignant_lesions     0.9310    0.9205    0.9257       176
body_of_stomach_normal     0.9841    0.9688    0.9764       192

              accuracy                         0.9410       576
             macro avg     0.9420    0.9406    0.9412       576
          weighted avg     0.9414    0.9410    0.9411       576

```