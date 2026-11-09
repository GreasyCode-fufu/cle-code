# starnet_s2_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce4m_seed44/starnet_s2_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9426 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9916 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9426 |
| Balanced Acc | 0.9424 |
| Macro-Sensitivity | 0.9424 |
| Macro-Specificity | 0.9712 |
| Macro-Precision | 0.9429 |
| AUC-PR (Macro) | 0.9839 |
| Cohen's Kappa | 0.9138 |
| MCC | 0.9139 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4943.5 (0.20 ms/张) |
| End-to-End FPS | 7.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9592 | 0.9282 | 0.9305 | 0.9619 | 0.9876 | 0.9801 | 208 |
| Malignant_lesions | 0.9205 | 0.9725 | 0.9364 | 0.9284 | 0.9653 | 0.9890 | 0.9751 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9818 | 0.9639 | 0.9689 | 0.9869 | 0.9982 | 0.9965 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=15 TN=353 FN=14
- **Malignant_lesions**: TP=162 FP=11 TN=389 FN=14
- **body_of_stomach_normal**: TP=187 FP=7 TN=377 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 11 | 3 |
| Malignant_lesions | 10 | 162 | 4 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9282    0.9327    0.9305       208
     Malignant_lesions     0.9364    0.9205    0.9284       176
body_of_stomach_normal     0.9639    0.9740    0.9689       192

              accuracy                         0.9427       576
             macro avg     0.9429    0.9424    0.9426       576
          weighted avg     0.9426    0.9427    0.9426       576

```