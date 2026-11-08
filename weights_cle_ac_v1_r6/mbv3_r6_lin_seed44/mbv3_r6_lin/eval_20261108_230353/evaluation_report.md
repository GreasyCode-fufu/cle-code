# mbv3_r6_lin 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_lin_seed44/mbv3_r6_lin/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9381 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9918 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9387 |
| Balanced Acc | 0.9355 |
| Macro-Sensitivity | 0.9355 |
| Macro-Specificity | 0.9687 |
| Macro-Precision | 0.9459 |
| AUC-PR (Macro) | 0.9851 |
| Cohen's Kappa | 0.9083 |
| MCC | 0.9107 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 4462.6 (0.22 ms/张) |
| End-to-End FPS | 228.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9293 | 0.8870 | 0.9315 | 0.9884 | 0.9893 | 0.9806 | 208 |
| Malignant_lesions | 0.8466 | 0.9950 | 0.9868 | 0.9113 | 0.9365 | 0.9875 | 0.9770 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9818 | 0.9641 | 0.9716 | 0.9895 | 0.9988 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=26 TN=342 FN=4
- **Malignant_lesions**: TP=149 FP=2 TN=398 FN=27
- **body_of_stomach_normal**: TP=188 FP=7 TN=377 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 2 | 2 |
| Malignant_lesions | 22 | 149 | 5 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8870    0.9808    0.9315       208
     Malignant_lesions     0.9868    0.8466    0.9113       176
body_of_stomach_normal     0.9641    0.9792    0.9716       192

              accuracy                         0.9392       576
             macro avg     0.9459    0.9355    0.9381       576
          weighted avg     0.9432    0.9392    0.9387       576

```