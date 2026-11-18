# starnet_s2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_seed43/starnet_s2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9284 |
| Accuracy | 0.9288 |
| Macro-AUROC (OvR) | 0.9896 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9289 |
| Balanced Acc | 0.9257 |
| Macro-Sensitivity | 0.9257 |
| Macro-Specificity | 0.9636 |
| Macro-Precision | 0.9347 |
| AUC-PR (Macro) | 0.9795 |
| Cohen's Kappa | 0.8927 |
| MCC | 0.8945 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3579.1 (0.28 ms/张) |
| End-to-End FPS | 20.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9185 | 0.8707 | 0.9182 | 0.9826 | 0.9850 | 0.9793 | 208 |
| Malignant_lesions | 0.8580 | 0.9800 | 0.9497 | 0.9015 | 0.9400 | 0.9868 | 0.9649 | 176 |
| body_of_stomach_normal | 0.9479 | 0.9922 | 0.9838 | 0.9655 | 0.9744 | 0.9970 | 0.9942 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=30 TN=338 FN=6
- **Malignant_lesions**: TP=151 FP=8 TN=392 FN=25
- **body_of_stomach_normal**: TP=182 FP=3 TN=381 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 23 | 151 | 2 |
| body_of_stomach_normal | 7 | 3 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8707    0.9712    0.9182       208
     Malignant_lesions     0.9497    0.8580    0.9015       176
body_of_stomach_normal     0.9838    0.9479    0.9655       192

              accuracy                         0.9288       576
             macro avg     0.9347    0.9257    0.9284       576
          weighted avg     0.9325    0.9288    0.9289       576

```