# mbv3_crt 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_crt_seed43/mbv3_crt/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9383 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9915 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9387 |
| Balanced Acc | 0.9361 |
| Macro-Sensitivity | 0.9361 |
| Macro-Specificity | 0.9688 |
| Macro-Precision | 0.9447 |
| AUC-PR (Macro) | 0.9823 |
| Cohen's Kappa | 0.9084 |
| MCC | 0.9102 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 2420.3 (0.41 ms/张) |
| End-to-End FPS | 8.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9321 | 0.8894 | 0.9263 | 0.9800 | 0.9866 | 0.9730 | 208 |
| Malignant_lesions | 0.8523 | 0.9925 | 0.9804 | 0.9119 | 0.9385 | 0.9887 | 0.9754 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9818 | 0.9645 | 0.9769 | 0.9947 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=25 TN=343 FN=7
- **Malignant_lesions**: TP=150 FP=3 TN=397 FN=26
- **body_of_stomach_normal**: TP=190 FP=7 TN=377 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 3 | 4 |
| Malignant_lesions | 23 | 150 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8894    0.9663    0.9263       208
     Malignant_lesions     0.9804    0.8523    0.9119       176
body_of_stomach_normal     0.9645    0.9896    0.9769       192

              accuracy                         0.9392       576
             macro avg     0.9447    0.9361    0.9383       576
          weighted avg     0.9422    0.9392    0.9387       576

```