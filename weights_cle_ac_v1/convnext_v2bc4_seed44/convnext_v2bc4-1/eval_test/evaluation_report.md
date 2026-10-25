# convnext_v2bc4 评估报告

- 权重: `weights_cle_ac_v1/convnext_v2bc4_seed44/convnext_v2bc4-1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9598 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9600 |
| Balanced Acc | 0.9607 |
| Macro-Sensitivity | 0.9607 |
| Macro-Specificity | 0.9802 |
| Macro-Precision | 0.9593 |
| AUC-PR (Macro) | 0.9935 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9402 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.42M |
| Pure-Inference FPS | 1631.6 (0.61 ms/张) |
| End-to-End FPS | 214.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9810 | 0.9652 | 0.9487 | 0.9627 | 0.9951 | 0.9918 | 208 |
| Malignant_lesions | 0.9545 | 0.9675 | 0.9282 | 0.9412 | 0.9797 | 0.9952 | 0.9892 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9922 | 0.9845 | 0.9896 | 0.9974 | 0.9998 | 0.9995 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=7 TN=361 FN=14
- **Malignant_lesions**: TP=168 FP=13 TN=387 FN=8
- **body_of_stomach_normal**: TP=191 FP=3 TN=381 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 13 | 1 |
| Malignant_lesions | 6 | 168 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9652    0.9327    0.9487       208
     Malignant_lesions     0.9282    0.9545    0.9412       176
body_of_stomach_normal     0.9845    0.9948    0.9896       192

              accuracy                         0.9601       576
             macro avg     0.9593    0.9607    0.9598       576
          weighted avg     0.9603    0.9601    0.9600       576

```