# convnext_v2cam2 评估报告

- 权重: `weights_cle_ac_v1_c/convnext_v2cam2_seed42/convnext_v2cam2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9600 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9956 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9600 |
| Balanced Acc | 0.9608 |
| Macro-Sensitivity | 0.9608 |
| Macro-Specificity | 0.9802 |
| Macro-Precision | 0.9594 |
| AUC-PR (Macro) | 0.9918 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9402 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 414.4 (2.41 ms/张) |
| End-to-End FPS | 16.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9810 | 0.9652 | 0.9487 | 0.9627 | 0.9939 | 0.9892 | 208 |
| Malignant_lesions | 0.9602 | 0.9700 | 0.9337 | 0.9468 | 0.9823 | 0.9935 | 0.9871 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9896 | 0.9794 | 0.9845 | 0.9948 | 0.9995 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=7 TN=361 FN=14
- **Malignant_lesions**: TP=169 FP=12 TN=388 FN=7
- **body_of_stomach_normal**: TP=190 FP=4 TN=380 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 12 | 2 |
| Malignant_lesions | 5 | 169 | 2 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9652    0.9327    0.9487       208
     Malignant_lesions     0.9337    0.9602    0.9468       176
body_of_stomach_normal     0.9794    0.9896    0.9845       192

              accuracy                         0.9601       576
             macro avg     0.9594    0.9608    0.9600       576
          weighted avg     0.9603    0.9601    0.9600       576

```