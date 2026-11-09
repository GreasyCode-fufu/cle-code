# starnet_s2_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce4m_seed46/starnet_s2_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9381 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9914 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9387 |
| Balanced Acc | 0.9369 |
| Macro-Sensitivity | 0.9369 |
| Macro-Specificity | 0.9691 |
| Macro-Precision | 0.9412 |
| AUC-PR (Macro) | 0.9833 |
| Cohen's Kappa | 0.9085 |
| MCC | 0.9093 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3562.8 (0.28 ms/张) |
| End-to-End FPS | 7.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9484 | 0.9124 | 0.9318 | 0.9721 | 0.9861 | 0.9797 | 208 |
| Malignant_lesions | 0.8693 | 0.9825 | 0.9563 | 0.9107 | 0.9447 | 0.9891 | 0.9723 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9766 | 0.9548 | 0.9719 | 0.9947 | 0.9990 | 0.9980 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=19 TN=349 FN=10
- **Malignant_lesions**: TP=153 FP=7 TN=393 FN=23
- **body_of_stomach_normal**: TP=190 FP=9 TN=375 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 17 | 153 | 6 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9124    0.9519    0.9318       208
     Malignant_lesions     0.9563    0.8693    0.9107       176
body_of_stomach_normal     0.9548    0.9896    0.9719       192

              accuracy                         0.9392       576
             macro avg     0.9412    0.9369    0.9381       576
          weighted avg     0.9399    0.9392    0.9387       576

```