# convnext_wds_wtb_s234 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wtb_s234_seed42/convnext_wds_wtb_s234/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9447 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9958 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9446 |
| Balanced Acc | 0.9432 |
| Macro-Sensitivity | 0.9432 |
| Macro-Specificity | 0.9717 |
| Macro-Precision | 0.9475 |
| AUC-PR (Macro) | 0.9922 |
| Cohen's Kappa | 0.9163 |
| MCC | 0.9170 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.56G |
| FLOPs | 9.12G |
| Params | 29.24M |
| Pure-Inference FPS | 723.1 (1.38 ms/张) |
| End-to-End FPS | 215.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9429 | 0.9054 | 0.9349 | 0.9802 | 0.9940 | 0.9906 | 208 |
| Malignant_lesions | 0.9205 | 0.9825 | 0.9586 | 0.9391 | 0.9656 | 0.9949 | 0.9890 | 176 |
| body_of_stomach_normal | 0.9427 | 0.9896 | 0.9784 | 0.9602 | 0.9719 | 0.9984 | 0.9970 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=21 TN=347 FN=7
- **Malignant_lesions**: TP=162 FP=7 TN=393 FN=14
- **body_of_stomach_normal**: TP=181 FP=4 TN=380 FN=11

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 11 | 162 | 3 |
| body_of_stomach_normal | 10 | 1 | 181 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9054    0.9663    0.9349       208
     Malignant_lesions     0.9586    0.9205    0.9391       176
body_of_stomach_normal     0.9784    0.9427    0.9602       192

              accuracy                         0.9444       576
             macro avg     0.9475    0.9432    0.9447       576
          weighted avg     0.9460    0.9444    0.9446       576

```