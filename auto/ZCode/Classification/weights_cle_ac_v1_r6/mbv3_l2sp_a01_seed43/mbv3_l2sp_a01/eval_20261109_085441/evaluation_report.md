# mbv3_l2sp_a01 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_l2sp_a01_seed43/mbv3_l2sp_a01/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9387 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9929 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9389 |
| Balanced Acc | 0.9387 |
| Macro-Sensitivity | 0.9387 |
| Macro-Specificity | 0.9695 |
| Macro-Precision | 0.9390 |
| AUC-PR (Macro) | 0.9861 |
| Cohen's Kappa | 0.9086 |
| MCC | 0.9087 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3129.3 (0.32 ms/张) |
| End-to-End FPS | 33.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9592 | 0.9275 | 0.9253 | 0.9566 | 0.9885 | 0.9806 | 208 |
| Malignant_lesions | 0.9034 | 0.9700 | 0.9298 | 0.9164 | 0.9580 | 0.9908 | 0.9788 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9792 | 0.9596 | 0.9744 | 0.9947 | 0.9994 | 0.9989 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=15 TN=353 FN=16
- **Malignant_lesions**: TP=159 FP=12 TN=388 FN=17
- **body_of_stomach_normal**: TP=190 FP=8 TN=376 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 11 | 5 |
| Malignant_lesions | 14 | 159 | 3 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9275    0.9231    0.9253       208
     Malignant_lesions     0.9298    0.9034    0.9164       176
body_of_stomach_normal     0.9596    0.9896    0.9744       192

              accuracy                         0.9392       576
             macro avg     0.9390    0.9387    0.9387       576
          weighted avg     0.9389    0.9392    0.9389       576

```