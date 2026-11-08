# convnext_v6c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v6c_seed43/convnext_v6c/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9545 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9547 |
| Balanced Acc | 0.9553 |
| Macro-Sensitivity | 0.9553 |
| Macro-Specificity | 0.9776 |
| Macro-Precision | 0.9542 |
| AUC-PR (Macro) | 0.9903 |
| Cohen's Kappa | 0.9322 |
| MCC | 0.9325 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 5.78G |
| FLOPs | 11.57G |
| Params | 30.61M |
| Pure-Inference FPS | 405.2 (2.47 ms/张) |
| End-to-End FPS | 171.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9837 | 0.9698 | 0.9484 | 0.9602 | 0.9930 | 0.9891 | 208 |
| Malignant_lesions | 0.9432 | 0.9725 | 0.9379 | 0.9405 | 0.9749 | 0.9926 | 0.9836 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9766 | 0.9550 | 0.9745 | 0.9973 | 0.9992 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=6 TN=362 FN=15
- **Malignant_lesions**: TP=166 FP=11 TN=389 FN=10
- **body_of_stomach_normal**: TP=191 FP=9 TN=375 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 11 | 4 |
| Malignant_lesions | 5 | 166 | 5 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9698    0.9279    0.9484       208
     Malignant_lesions     0.9379    0.9432    0.9405       176
body_of_stomach_normal     0.9550    0.9948    0.9745       192

              accuracy                         0.9549       576
             macro avg     0.9542    0.9553    0.9545       576
          weighted avg     0.9551    0.9549    0.9547       576

```