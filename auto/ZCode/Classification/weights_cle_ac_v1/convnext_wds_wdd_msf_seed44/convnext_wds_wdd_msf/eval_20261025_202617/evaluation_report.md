# convnext_wds_wdd_msf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_wdd_msf_seed44/convnext_wds_wdd_msf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9379 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9932 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9386 |
| Balanced Acc | 0.9359 |
| Macro-Sensitivity | 0.9359 |
| Macro-Specificity | 0.9689 |
| Macro-Precision | 0.9433 |
| AUC-PR (Macro) | 0.9873 |
| Cohen's Kappa | 0.9084 |
| MCC | 0.9099 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.68G |
| FLOPs | 9.36G |
| Params | 29.80M |
| Pure-Inference FPS | 453.0 (2.21 ms/张) |
| End-to-End FPS | 186.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9375 | 0.8978 | 0.9330 | 0.9829 | 0.9914 | 0.9815 | 208 |
| Malignant_lesions | 0.8523 | 0.9875 | 0.9677 | 0.9063 | 0.9382 | 0.9892 | 0.9821 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9818 | 0.9643 | 0.9742 | 0.9921 | 0.9991 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=23 TN=345 FN=6
- **Malignant_lesions**: TP=150 FP=5 TN=395 FN=26
- **body_of_stomach_normal**: TP=189 FP=7 TN=377 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 20 | 150 | 6 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8978    0.9712    0.9330       208
     Malignant_lesions     0.9677    0.8523    0.9063       176
body_of_stomach_normal     0.9643    0.9844    0.9742       192

              accuracy                         0.9392       576
             macro avg     0.9433    0.9359    0.9379       576
          weighted avg     0.9413    0.9392    0.9386       576

```