# fast_t0_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_dce4m_seed46/fast_t0_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9301 |
| Accuracy | 0.9306 |
| Macro-AUROC (OvR) | 0.9937 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9303 |
| Balanced Acc | 0.9277 |
| Macro-Sensitivity | 0.9277 |
| Macro-Specificity | 0.9644 |
| Macro-Precision | 0.9359 |
| AUC-PR (Macro) | 0.9880 |
| Cohen's Kappa | 0.8953 |
| MCC | 0.8969 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 7309.4 (0.14 ms/张) |
| End-to-End FPS | 7.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9266 | 0.8811 | 0.9195 | 0.9771 | 0.9897 | 0.9834 | 208 |
| Malignant_lesions | 0.8580 | 0.9875 | 0.9679 | 0.9096 | 0.9405 | 0.9927 | 0.9832 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9792 | 0.9585 | 0.9610 | 0.9817 | 0.9986 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=27 TN=341 FN=8
- **Malignant_lesions**: TP=151 FP=5 TN=395 FN=25
- **body_of_stomach_normal**: TP=185 FP=8 TN=376 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 5 | 3 |
| Malignant_lesions | 20 | 151 | 5 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8811    0.9615    0.9195       208
     Malignant_lesions     0.9679    0.8580    0.9096       176
body_of_stomach_normal     0.9585    0.9635    0.9610       192

              accuracy                         0.9306       576
             macro avg     0.9359    0.9277    0.9301       576
          weighted avg     0.9334    0.9306    0.9303       576

```