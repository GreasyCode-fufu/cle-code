# convnext_wds_wdd_msf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wdd_msf_seed42/convnext_wds_wdd_msf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9568 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9962 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9567 |
| Balanced Acc | 0.9565 |
| Macro-Sensitivity | 0.9565 |
| Macro-Specificity | 0.9782 |
| Macro-Precision | 0.9572 |
| AUC-PR (Macro) | 0.9930 |
| Cohen's Kappa | 0.9347 |
| MCC | 0.9349 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.68G |
| FLOPs | 9.36G |
| Params | 29.80M |
| Pure-Inference FPS | 772.6 (1.29 ms/张) |
| End-to-End FPS | 222.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9647 | 0.9387 | 0.9476 | 0.9753 | 0.9941 | 0.9897 | 208 |
| Malignant_lesions | 0.9545 | 0.9750 | 0.9438 | 0.9492 | 0.9799 | 0.9955 | 0.9909 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9948 | 0.9892 | 0.9735 | 0.9795 | 0.9991 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=13 TN=355 FN=9
- **Malignant_lesions**: TP=168 FP=10 TN=390 FN=8
- **body_of_stomach_normal**: TP=184 FP=2 TN=382 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 9 | 0 |
| Malignant_lesions | 6 | 168 | 2 |
| body_of_stomach_normal | 7 | 1 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9387    0.9567    0.9476       208
     Malignant_lesions     0.9438    0.9545    0.9492       176
body_of_stomach_normal     0.9892    0.9583    0.9735       192

              accuracy                         0.9566       576
             macro avg     0.9572    0.9565    0.9568       576
          weighted avg     0.9571    0.9566    0.9567       576

```