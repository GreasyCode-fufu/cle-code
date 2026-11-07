# convnext_v2ccw 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2ccw_seed42/convnext_v2ccw/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9635 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9636 |
| Balanced Acc | 0.9623 |
| Macro-Sensitivity | 0.9623 |
| Macro-Specificity | 0.9814 |
| Macro-Precision | 0.9653 |
| AUC-PR (Macro) | 0.9940 |
| Cohen's Kappa | 0.9451 |
| MCC | 0.9454 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 454.9 (2.20 ms/张) |
| End-to-End FPS | 180.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9620 | 0.9355 | 0.9553 | 0.9861 | 0.9954 | 0.9923 | 208 |
| Malignant_lesions | 0.9318 | 0.9900 | 0.9762 | 0.9535 | 0.9706 | 0.9958 | 0.9911 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=14 TN=354 FN=5
- **Malignant_lesions**: TP=164 FP=4 TN=396 FN=12
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 4 | 1 |
| Malignant_lesions | 10 | 164 | 2 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9355    0.9760    0.9553       208
     Malignant_lesions     0.9762    0.9318    0.9535       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9635       576
             macro avg     0.9653    0.9623    0.9635       576
          weighted avg     0.9642    0.9635    0.9636       576

```