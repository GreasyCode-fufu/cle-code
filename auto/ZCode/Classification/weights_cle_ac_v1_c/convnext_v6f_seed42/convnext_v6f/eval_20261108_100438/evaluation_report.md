# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v6f_seed42/convnext_v6f/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9599 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9959 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9599 |
| Balanced Acc | 0.9613 |
| Macro-Sensitivity | 0.9613 |
| Macro-Specificity | 0.9803 |
| Macro-Precision | 0.9595 |
| AUC-PR (Macro) | 0.9927 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9406 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 441.6 (2.26 ms/张) |
| End-to-End FPS | 185.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9918 | 0.9846 | 0.9529 | 0.9580 | 0.9939 | 0.9917 | 208 |
| Malignant_lesions | 0.9659 | 0.9700 | 0.9341 | 0.9497 | 0.9848 | 0.9948 | 0.9884 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9792 | 0.9598 | 0.9770 | 0.9973 | 0.9990 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=3 TN=365 FN=16
- **Malignant_lesions**: TP=170 FP=12 TN=388 FN=6
- **body_of_stomach_normal**: TP=191 FP=8 TN=376 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 12 | 4 |
| Malignant_lesions | 2 | 170 | 4 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9846    0.9231    0.9529       208
     Malignant_lesions     0.9341    0.9659    0.9497       176
body_of_stomach_normal     0.9598    0.9948    0.9770       192

              accuracy                         0.9601       576
             macro avg     0.9595    0.9613    0.9599       576
          weighted avg     0.9609    0.9601    0.9599       576

```