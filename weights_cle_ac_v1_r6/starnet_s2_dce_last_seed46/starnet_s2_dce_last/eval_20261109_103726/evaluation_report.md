# starnet_s2_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce_last_seed46/starnet_s2_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9584 |
| Accuracy | 0.9672 |
| Macro-AUROC (OvR) | 0.9945 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9668 |
| Balanced Acc | 0.9510 |
| Macro-Sensitivity | 0.9510 |
| Macro-Specificity | 0.9815 |
| Macro-Precision | 0.9674 |
| AUC-PR (Macro) | 0.9818 |
| Cohen's Kappa | 0.9475 |
| MCC | 0.9479 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4204.7 (0.24 ms/张) |
| End-to-End FPS | 8.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9569 | 0.9502 | 0.9646 | 0.9823 | 0.9919 | 0.9933 | 195 |
| Malignant_lesions | 0.8734 | 0.9914 | 0.9583 | 0.9139 | 0.9718 | 0.9917 | 0.9519 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=10 TN=222 FN=4
- **Malignant_lesions**: TP=69 FP=3 TN=345 FN=10
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 3 | 1 |
| Malignant_lesions | 10 | 69 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9502    0.9795    0.9646       195
     Malignant_lesions     0.9583    0.8734    0.9139        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9672       427
             macro avg     0.9674    0.9510    0.9584       427
          weighted avg     0.9672    0.9672    0.9668       427

```