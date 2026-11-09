# efficientvim_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/efficientvim_m1_seed44/efficientvim_m1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9436 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9932 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9441 |
| Balanced Acc | 0.9432 |
| Macro-Sensitivity | 0.9432 |
| Macro-Specificity | 0.9720 |
| Macro-Precision | 0.9450 |
| AUC-PR (Macro) | 0.9873 |
| Cohen's Kappa | 0.9164 |
| MCC | 0.9168 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 234.46M |
| FLOPs | 468.92M |
| Params | 5.71M |
| Pure-Inference FPS | 5754.7 (0.17 ms/张) |
| End-to-End FPS | 248.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9647 | 0.9378 | 0.9400 | 0.9673 | 0.9891 | 0.9850 | 208 |
| Malignant_lesions | 0.8977 | 0.9800 | 0.9518 | 0.9240 | 0.9561 | 0.9926 | 0.9821 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9714 | 0.9453 | 0.9669 | 0.9947 | 0.9978 | 0.9947 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=13 TN=355 FN=12
- **Malignant_lesions**: TP=158 FP=8 TN=392 FN=18
- **body_of_stomach_normal**: TP=190 FP=11 TN=373 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 8 | 4 |
| Malignant_lesions | 11 | 158 | 7 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9378    0.9423    0.9400       208
     Malignant_lesions     0.9518    0.8977    0.9240       176
body_of_stomach_normal     0.9453    0.9896    0.9669       192

              accuracy                         0.9444       576
             macro avg     0.9450    0.9432    0.9436       576
          weighted avg     0.9446    0.9444    0.9441       576

```