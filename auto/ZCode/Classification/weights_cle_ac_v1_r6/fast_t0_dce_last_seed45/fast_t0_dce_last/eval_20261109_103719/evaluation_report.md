# fast_t0_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_dce_last_seed45/fast_t0_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9061 |
| Accuracy | 0.9080 |
| Macro-AUROC (OvR) | 0.9900 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9071 |
| Balanced Acc | 0.9028 |
| Macro-Sensitivity | 0.9028 |
| Macro-Specificity | 0.9527 |
| Macro-Precision | 0.9191 |
| AUC-PR (Macro) | 0.9801 |
| Cohen's Kappa | 0.8610 |
| MCC | 0.8655 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 8156.5 (0.12 ms/张) |
| End-to-End FPS | 7.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.8886 | 0.8306 | 0.8933 | 0.9790 | 0.9862 | 0.9770 | 208 |
| Malignant_lesions | 0.7784 | 0.9850 | 0.9580 | 0.8589 | 0.9099 | 0.9851 | 0.9661 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9844 | 0.9686 | 0.9661 | 0.9818 | 0.9986 | 0.9971 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=41 TN=327 FN=7
- **Malignant_lesions**: TP=137 FP=6 TN=394 FN=39
- **body_of_stomach_normal**: TP=185 FP=6 TN=378 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 34 | 137 | 5 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8306    0.9663    0.8933       208
     Malignant_lesions     0.9580    0.7784    0.8589       176
body_of_stomach_normal     0.9686    0.9635    0.9661       192

              accuracy                         0.9080       576
             macro avg     0.9191    0.9028    0.9061       576
          weighted avg     0.9155    0.9080    0.9071       576

```