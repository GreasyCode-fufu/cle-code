# convnext_cmp_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_cmp_dsd_seed43/convnext_cmp_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9542 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9954 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9547 |
| Balanced Acc | 0.9527 |
| Macro-Sensitivity | 0.9527 |
| Macro-Specificity | 0.9770 |
| Macro-Precision | 0.9572 |
| AUC-PR (Macro) | 0.9917 |
| Cohen's Kappa | 0.9320 |
| MCC | 0.9327 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 28.23M |
| Pure-Inference FPS | 605.1 (1.65 ms/张) |
| End-to-End FPS | 203.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9511 | 0.9186 | 0.9464 | 0.9859 | 0.9951 | 0.9917 | 208 |
| Malignant_lesions | 0.8977 | 0.9850 | 0.9634 | 0.9294 | 0.9563 | 0.9918 | 0.9847 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9948 | 0.9895 | 0.9869 | 0.9922 | 0.9994 | 0.9989 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=18 TN=350 FN=5
- **Malignant_lesions**: TP=158 FP=6 TN=394 FN=18
- **body_of_stomach_normal**: TP=189 FP=2 TN=382 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 5 | 0 |
| Malignant_lesions | 16 | 158 | 2 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9186    0.9760    0.9464       208
     Malignant_lesions     0.9634    0.8977    0.9294       176
body_of_stomach_normal     0.9895    0.9844    0.9869       192

              accuracy                         0.9549       576
             macro avg     0.9572    0.9527    0.9542       576
          weighted avg     0.9559    0.9549    0.9547       576

```