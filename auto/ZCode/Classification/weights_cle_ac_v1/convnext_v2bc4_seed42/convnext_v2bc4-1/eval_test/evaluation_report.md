# convnext_v2bc4 评估报告

- 权重: `weights_cle_ac_v1/convnext_v2bc4_seed42/convnext_v2bc4-1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9418 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9950 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9413 |
| Balanced Acc | 0.9409 |
| Macro-Sensitivity | 0.9409 |
| Macro-Specificity | 0.9700 |
| Macro-Precision | 0.9442 |
| AUC-PR (Macro) | 0.9909 |
| Cohen's Kappa | 0.9112 |
| MCC | 0.9119 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.42M |
| Pure-Inference FPS | 1624.0 (0.62 ms/张) |
| End-to-End FPS | 229.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9402 | 0.9000 | 0.9252 | 0.9719 | 0.9923 | 0.9882 | 208 |
| Malignant_lesions | 0.9489 | 0.9725 | 0.9382 | 0.9435 | 0.9774 | 0.9952 | 0.9889 | 176 |
| body_of_stomach_normal | 0.9219 | 0.9974 | 0.9944 | 0.9568 | 0.9623 | 0.9975 | 0.9956 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=22 TN=346 FN=10
- **Malignant_lesions**: TP=167 FP=11 TN=389 FN=9
- **body_of_stomach_normal**: TP=177 FP=1 TN=383 FN=15

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 9 | 167 | 0 |
| body_of_stomach_normal | 13 | 2 | 177 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9000    0.9519    0.9252       208
     Malignant_lesions     0.9382    0.9489    0.9435       176
body_of_stomach_normal     0.9944    0.9219    0.9568       192

              accuracy                         0.9410       576
             macro avg     0.9442    0.9409    0.9418       576
          weighted avg     0.9431    0.9410    0.9413       576

```