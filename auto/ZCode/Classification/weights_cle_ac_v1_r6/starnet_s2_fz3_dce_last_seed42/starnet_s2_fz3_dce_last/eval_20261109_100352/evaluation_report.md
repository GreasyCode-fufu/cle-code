# starnet_s2_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce_last_seed42/starnet_s2_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9184 |
| Accuracy | 0.9201 |
| Macro-AUROC (OvR) | 0.9861 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9193 |
| Balanced Acc | 0.9174 |
| Macro-Sensitivity | 0.9174 |
| Macro-Specificity | 0.9596 |
| Macro-Precision | 0.9214 |
| AUC-PR (Macro) | 0.9730 |
| Cohen's Kappa | 0.8797 |
| MCC | 0.8806 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4911.7 (0.20 ms/张) |
| End-to-End FPS | 6.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9348 | 0.8899 | 0.9108 | 0.9609 | 0.9779 | 0.9705 | 208 |
| Malignant_lesions | 0.8352 | 0.9700 | 0.9245 | 0.8776 | 0.9305 | 0.9818 | 0.9515 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9740 | 0.9497 | 0.9668 | 0.9920 | 0.9985 | 0.9969 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=24 TN=344 FN=14
- **Malignant_lesions**: TP=147 FP=12 TN=388 FN=29
- **body_of_stomach_normal**: TP=189 FP=10 TN=374 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 11 | 3 |
| Malignant_lesions | 22 | 147 | 7 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8899    0.9327    0.9108       208
     Malignant_lesions     0.9245    0.8352    0.8776       176
body_of_stomach_normal     0.9497    0.9844    0.9668       192

              accuracy                         0.9201       576
             macro avg     0.9214    0.9174    0.9184       576
          weighted avg     0.9204    0.9201    0.9193       576

```