# convnext_dph_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_dsd_seed42/convnext_dph_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9480 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9947 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9483 |
| Balanced Acc | 0.9497 |
| Macro-Sensitivity | 0.9497 |
| Macro-Specificity | 0.9745 |
| Macro-Precision | 0.9486 |
| AUC-PR (Macro) | 0.9906 |
| Cohen's Kappa | 0.9219 |
| MCC | 0.9229 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 28.68M |
| Pure-Inference FPS | 882.8 (1.13 ms/张) |
| End-to-End FPS | 229.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9135 | 0.9810 | 0.9645 | 0.9383 | 0.9525 | 0.9917 | 0.9880 | 208 |
| Malignant_lesions | 0.9773 | 0.9450 | 0.8866 | 0.9297 | 0.9895 | 0.9925 | 0.9843 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9974 | 0.9946 | 0.9761 | 0.9795 | 0.9997 | 0.9995 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=7 TN=361 FN=18
- **Malignant_lesions**: TP=172 FP=22 TN=378 FN=4
- **body_of_stomach_normal**: TP=184 FP=1 TN=383 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 17 | 1 |
| Malignant_lesions | 4 | 172 | 0 |
| body_of_stomach_normal | 3 | 5 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9645    0.9135    0.9383       208
     Malignant_lesions     0.8866    0.9773    0.9297       176
body_of_stomach_normal     0.9946    0.9583    0.9761       192

              accuracy                         0.9479       576
             macro avg     0.9486    0.9497    0.9480       576
          weighted avg     0.9507    0.9479    0.9483       576

```