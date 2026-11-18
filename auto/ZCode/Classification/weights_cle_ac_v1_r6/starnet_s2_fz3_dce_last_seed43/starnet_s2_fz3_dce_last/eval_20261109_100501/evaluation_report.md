# starnet_s2_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce_last_seed43/starnet_s2_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9733 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9985 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9787 |
| Balanced Acc | 0.9696 |
| Macro-Sensitivity | 0.9696 |
| Macro-Specificity | 0.9889 |
| Macro-Precision | 0.9775 |
| AUC-PR (Macro) | 0.9958 |
| Cohen's Kappa | 0.9664 |
| MCC | 0.9665 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 2461.1 (0.41 ms/张) |
| End-to-End FPS | 7.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9871 | 0.9846 | 0.9846 | 0.9871 | 0.9985 | 0.9982 | 195 |
| Malignant_lesions | 0.9241 | 0.9943 | 0.9733 | 0.9481 | 0.9830 | 0.9969 | 0.9891 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9854 | 0.9745 | 0.9871 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=3 TN=229 FN=3
- **Malignant_lesions**: TP=73 FP=2 TN=346 FN=6
- **body_of_stomach_normal**: TP=153 FP=4 TN=270 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 2 | 1 |
| Malignant_lesions | 3 | 73 | 3 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9846    0.9846    0.9846       195
     Malignant_lesions     0.9733    0.9241    0.9481        79
body_of_stomach_normal     0.9745    1.0000    0.9871       153

              accuracy                         0.9789       427
             macro avg     0.9775    0.9696    0.9733       427
          weighted avg     0.9789    0.9789    0.9787       427

```