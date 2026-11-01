# convnext_dph_am2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_am2_seed43/convnext_dph_am2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9495 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9956 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9496 |
| Balanced Acc | 0.9505 |
| Macro-Sensitivity | 0.9505 |
| Macro-Specificity | 0.9750 |
| Macro-Precision | 0.9489 |
| AUC-PR (Macro) | 0.9916 |
| Cohen's Kappa | 0.9244 |
| MCC | 0.9246 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 679.9 (1.47 ms/张) |
| End-to-End FPS | 208.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9183 | 0.9755 | 0.9550 | 0.9363 | 0.9548 | 0.9935 | 0.9898 | 208 |
| Malignant_lesions | 0.9489 | 0.9600 | 0.9126 | 0.9304 | 0.9771 | 0.9938 | 0.9861 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9896 | 0.9793 | 0.9818 | 0.9922 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=9 TN=359 FN=17
- **Malignant_lesions**: TP=167 FP=16 TN=384 FN=9
- **body_of_stomach_normal**: TP=189 FP=4 TN=380 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 15 | 2 |
| Malignant_lesions | 7 | 167 | 2 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9550    0.9183    0.9363       208
     Malignant_lesions     0.9126    0.9489    0.9304       176
body_of_stomach_normal     0.9793    0.9844    0.9818       192

              accuracy                         0.9497       576
             macro avg     0.9489    0.9505    0.9495       576
          weighted avg     0.9501    0.9497    0.9496       576

```