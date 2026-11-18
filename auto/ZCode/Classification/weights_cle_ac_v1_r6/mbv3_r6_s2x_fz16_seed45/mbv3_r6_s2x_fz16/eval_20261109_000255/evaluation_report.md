# mbv3_r6_s2x_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2x_fz16_seed45/mbv3_r6_s2x_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9473 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9934 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9476 |
| Balanced Acc | 0.9464 |
| Macro-Sensitivity | 0.9464 |
| Macro-Specificity | 0.9736 |
| Macro-Precision | 0.9493 |
| AUC-PR (Macro) | 0.9877 |
| Cohen's Kappa | 0.9216 |
| MCC | 0.9221 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.10M |
| FLOPs | 464.20M |
| Params | 3.39M |
| Pure-Inference FPS | 4496.4 (0.22 ms/张) |
| End-to-End FPS | 25.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9592 | 0.9296 | 0.9406 | 0.9725 | 0.9883 | 0.9810 | 208 |
| Malignant_lesions | 0.8977 | 0.9850 | 0.9634 | 0.9294 | 0.9563 | 0.9932 | 0.9845 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9766 | 0.9548 | 0.9719 | 0.9947 | 0.9987 | 0.9975 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=15 TN=353 FN=10
- **Malignant_lesions**: TP=158 FP=6 TN=394 FN=18
- **body_of_stomach_normal**: TP=190 FP=9 TN=375 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 5 | 5 |
| Malignant_lesions | 14 | 158 | 4 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9296    0.9519    0.9406       208
     Malignant_lesions     0.9634    0.8977    0.9294       176
body_of_stomach_normal     0.9548    0.9896    0.9719       192

              accuracy                         0.9479       576
             macro avg     0.9493    0.9464    0.9473       576
          weighted avg     0.9483    0.9479    0.9476       576

```