# deit_small_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/deit_small_patch16_224_seed42/deit_small_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9319 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9902 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9320 |
| Balanced Acc | 0.9319 |
| Macro-Sensitivity | 0.9319 |
| Macro-Specificity | 0.9659 |
| Macro-Precision | 0.9323 |
| AUC-PR (Macro) | 0.9782 |
| Cohen's Kappa | 0.8982 |
| MCC | 0.8983 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.25G |
| FLOPs | 8.50G |
| Params | 21.59M |
| Pure-Inference FPS | 1319.5 (0.76 ms/张) |
| End-to-End FPS | 210.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9135 | 0.9538 | 0.9179 | 0.9157 | 0.9512 | 0.9846 | 0.9759 | 208 |
| Malignant_lesions | 0.8977 | 0.9700 | 0.9294 | 0.9133 | 0.9557 | 0.9874 | 0.9613 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9740 | 0.9497 | 0.9668 | 0.9920 | 0.9985 | 0.9972 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=17 TN=351 FN=18
- **Malignant_lesions**: TP=158 FP=12 TN=388 FN=18
- **body_of_stomach_normal**: TP=189 FP=10 TN=374 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 12 | 6 |
| Malignant_lesions | 14 | 158 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9179    0.9135    0.9157       208
     Malignant_lesions     0.9294    0.8977    0.9133       176
body_of_stomach_normal     0.9497    0.9844    0.9668       192

              accuracy                         0.9323       576
             macro avg     0.9323    0.9319    0.9319       576
          weighted avg     0.9320    0.9323    0.9320       576

```