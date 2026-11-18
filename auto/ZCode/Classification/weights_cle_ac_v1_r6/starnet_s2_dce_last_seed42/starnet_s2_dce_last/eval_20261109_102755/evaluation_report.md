# starnet_s2_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce_last_seed42/starnet_s2_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9538 |
| Accuracy | 0.9625 |
| Macro-AUROC (OvR) | 0.9957 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9626 |
| Balanced Acc | 0.9576 |
| Macro-Sensitivity | 0.9576 |
| Macro-Specificity | 0.9818 |
| Macro-Precision | 0.9503 |
| AUC-PR (Macro) | 0.9860 |
| Cohen's Kappa | 0.9406 |
| MCC | 0.9408 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3430.3 (0.29 ms/张) |
| End-to-End FPS | 8.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9487 | 0.9784 | 0.9737 | 0.9610 | 0.9578 | 0.9951 | 0.9945 | 195 |
| Malignant_lesions | 0.9241 | 0.9741 | 0.8902 | 0.9068 | 0.9826 | 0.9919 | 0.9634 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=185 FP=5 TN=227 FN=10
- **Malignant_lesions**: TP=73 FP=9 TN=339 FN=6
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 185 | 9 | 1 |
| Malignant_lesions | 5 | 73 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9737    0.9487    0.9610       195
     Malignant_lesions     0.8902    0.9241    0.9068        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9625       427
             macro avg     0.9503    0.9576    0.9538       427
          weighted avg     0.9631    0.9625    0.9626       427

```