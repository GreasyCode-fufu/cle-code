# starnet_s2_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce4m_seed43/starnet_s2_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9370 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9920 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9374 |
| Balanced Acc | 0.9348 |
| Macro-Sensitivity | 0.9348 |
| Macro-Specificity | 0.9681 |
| Macro-Precision | 0.9416 |
| AUC-PR (Macro) | 0.9847 |
| Cohen's Kappa | 0.9058 |
| MCC | 0.9070 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3537.8 (0.28 ms/张) |
| End-to-End FPS | 7.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9348 | 0.8938 | 0.9309 | 0.9829 | 0.9878 | 0.9834 | 208 |
| Malignant_lesions | 0.8750 | 0.9850 | 0.9625 | 0.9167 | 0.9471 | 0.9909 | 0.9760 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9844 | 0.9684 | 0.9634 | 0.9793 | 0.9972 | 0.9947 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=24 TN=344 FN=6
- **Malignant_lesions**: TP=154 FP=6 TN=394 FN=22
- **body_of_stomach_normal**: TP=184 FP=6 TN=378 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 17 | 154 | 5 |
| body_of_stomach_normal | 7 | 1 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8938    0.9712    0.9309       208
     Malignant_lesions     0.9625    0.8750    0.9167       176
body_of_stomach_normal     0.9684    0.9583    0.9634       192

              accuracy                         0.9375       576
             macro avg     0.9416    0.9348    0.9370       576
          weighted avg     0.9397    0.9375    0.9374       576

```