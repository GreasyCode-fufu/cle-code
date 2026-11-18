# convnext_cmp 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_cmp_seed42/convnext_cmp/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9441 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9935 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9444 |
| Balanced Acc | 0.9456 |
| Macro-Sensitivity | 0.9456 |
| Macro-Specificity | 0.9726 |
| Macro-Precision | 0.9438 |
| AUC-PR (Macro) | 0.9895 |
| Cohen's Kappa | 0.9166 |
| MCC | 0.9171 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 719.4 (1.39 ms/张) |
| End-to-End FPS | 210.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9087 | 0.9837 | 0.9692 | 0.9380 | 0.9501 | 0.9915 | 0.9889 | 208 |
| Malignant_lesions | 0.9489 | 0.9600 | 0.9126 | 0.9304 | 0.9771 | 0.9905 | 0.9825 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9740 | 0.9495 | 0.9641 | 0.9894 | 0.9985 | 0.9971 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=6 TN=362 FN=19
- **Malignant_lesions**: TP=167 FP=16 TN=384 FN=9
- **body_of_stomach_normal**: TP=188 FP=10 TN=374 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 15 | 4 |
| Malignant_lesions | 3 | 167 | 6 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9692    0.9087    0.9380       208
     Malignant_lesions     0.9126    0.9489    0.9304       176
body_of_stomach_normal     0.9495    0.9792    0.9641       192

              accuracy                         0.9444       576
             macro avg     0.9438    0.9456    0.9441       576
          weighted avg     0.9453    0.9444    0.9444       576

```