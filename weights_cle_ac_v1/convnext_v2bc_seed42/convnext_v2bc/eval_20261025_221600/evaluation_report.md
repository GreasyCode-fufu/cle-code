# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2bc_seed42/convnext_v2bc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9672 |
| Accuracy | 0.9670 |
| Macro-AUROC (OvR) | 0.9967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9671 |
| Balanced Acc | 0.9667 |
| Macro-Sensitivity | 0.9667 |
| Macro-Specificity | 0.9833 |
| Macro-Precision | 0.9678 |
| AUC-PR (Macro) | 0.9935 |
| Cohen's Kappa | 0.9504 |
| MCC | 0.9505 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 835.3 (1.20 ms/张) |
| End-to-End FPS | 219.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9701 | 0.9484 | 0.9596 | 0.9835 | 0.9955 | 0.9929 | 208 |
| Malignant_lesions | 0.9602 | 0.9850 | 0.9657 | 0.9630 | 0.9825 | 0.9957 | 0.9895 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9948 | 0.9894 | 0.9789 | 0.9845 | 0.9990 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=11 TN=357 FN=6
- **Malignant_lesions**: TP=169 FP=6 TN=394 FN=7
- **body_of_stomach_normal**: TP=186 FP=2 TN=382 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 6 | 0 |
| Malignant_lesions | 5 | 169 | 2 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9484    0.9712    0.9596       208
     Malignant_lesions     0.9657    0.9602    0.9630       176
body_of_stomach_normal     0.9894    0.9688    0.9789       192

              accuracy                         0.9670       576
             macro avg     0.9678    0.9667    0.9672       576
          weighted avg     0.9673    0.9670    0.9671       576

```