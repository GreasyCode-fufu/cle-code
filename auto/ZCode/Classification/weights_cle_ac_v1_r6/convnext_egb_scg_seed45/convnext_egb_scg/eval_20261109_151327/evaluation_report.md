# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_egb_scg_seed45/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9561 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9564 |
| Balanced Acc | 0.9553 |
| Macro-Sensitivity | 0.9553 |
| Macro-Specificity | 0.9780 |
| Macro-Precision | 0.9576 |
| AUC-PR (Macro) | 0.9935 |
| Cohen's Kappa | 0.9347 |
| MCC | 0.9350 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 310.9 (3.22 ms/张) |
| End-to-End FPS | 150.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9674 | 0.9434 | 0.9524 | 0.9780 | 0.9951 | 0.9914 | 208 |
| Malignant_lesions | 0.9148 | 0.9875 | 0.9699 | 0.9415 | 0.9634 | 0.9954 | 0.9902 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9792 | 0.9596 | 0.9744 | 0.9947 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=12 TN=356 FN=8
- **Malignant_lesions**: TP=161 FP=5 TN=395 FN=15
- **body_of_stomach_normal**: TP=190 FP=8 TN=376 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 5 | 3 |
| Malignant_lesions | 10 | 161 | 5 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9434    0.9615    0.9524       208
     Malignant_lesions     0.9699    0.9148    0.9415       176
body_of_stomach_normal     0.9596    0.9896    0.9744       192

              accuracy                         0.9566       576
             macro avg     0.9576    0.9553    0.9561       576
          weighted avg     0.9569    0.9566    0.9564       576

```