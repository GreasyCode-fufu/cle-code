# convnext_fgc_dsd_ri 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_dsd_ri_seed43/convnext_fgc_dsd_ri/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9512 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9962 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9514 |
| Balanced Acc | 0.9509 |
| Macro-Sensitivity | 0.9509 |
| Macro-Specificity | 0.9756 |
| Macro-Precision | 0.9514 |
| AUC-PR (Macro) | 0.9927 |
| Cohen's Kappa | 0.9269 |
| MCC | 0.9269 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 461.6 (2.17 ms/张) |
| End-to-End FPS | 184.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9620 | 0.9333 | 0.9378 | 0.9672 | 0.9947 | 0.9911 | 208 |
| Malignant_lesions | 0.9261 | 0.9700 | 0.9314 | 0.9288 | 0.9676 | 0.9944 | 0.9879 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9948 | 0.9895 | 0.9869 | 0.9922 | 0.9995 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=14 TN=354 FN=12
- **Malignant_lesions**: TP=163 FP=12 TN=388 FN=13
- **body_of_stomach_normal**: TP=189 FP=2 TN=382 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 11 | 1 |
| Malignant_lesions | 12 | 163 | 1 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9333    0.9423    0.9378       208
     Malignant_lesions     0.9314    0.9261    0.9288       176
body_of_stomach_normal     0.9895    0.9844    0.9869       192

              accuracy                         0.9514       576
             macro avg     0.9514    0.9509    0.9512       576
          weighted avg     0.9515    0.9514    0.9514       576

```