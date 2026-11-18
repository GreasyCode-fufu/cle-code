# mbv3_r6_s2_anc1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_anc1_seed42/mbv3_r6_s2_anc1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9434 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9920 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9440 |
| Balanced Acc | 0.9425 |
| Macro-Sensitivity | 0.9425 |
| Macro-Specificity | 0.9718 |
| Macro-Precision | 0.9458 |
| AUC-PR (Macro) | 0.9849 |
| Cohen's Kappa | 0.9163 |
| MCC | 0.9170 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 3808.1 (0.26 ms/张) |
| End-to-End FPS | 224.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9538 | 0.9209 | 0.9362 | 0.9723 | 0.9880 | 0.9773 | 208 |
| Malignant_lesions | 0.8807 | 0.9825 | 0.9568 | 0.9172 | 0.9493 | 0.9886 | 0.9785 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9792 | 0.9598 | 0.9770 | 0.9973 | 0.9994 | 0.9989 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=17 TN=351 FN=10
- **Malignant_lesions**: TP=155 FP=7 TN=393 FN=21
- **body_of_stomach_normal**: TP=191 FP=8 TN=376 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 6 | 4 |
| Malignant_lesions | 17 | 155 | 4 |
| body_of_stomach_normal | 0 | 1 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9209    0.9519    0.9362       208
     Malignant_lesions     0.9568    0.8807    0.9172       176
body_of_stomach_normal     0.9598    0.9948    0.9770       192

              accuracy                         0.9444       576
             macro avg     0.9458    0.9425    0.9434       576
          weighted avg     0.9448    0.9444    0.9440       576

```