# starnet_s2_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_seed46/starnet_s2_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9261 |
| Accuracy | 0.9271 |
| Macro-AUROC (OvR) | 0.9896 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9265 |
| Balanced Acc | 0.9254 |
| Macro-Sensitivity | 0.9254 |
| Macro-Specificity | 0.9631 |
| Macro-Precision | 0.9286 |
| AUC-PR (Macro) | 0.9797 |
| Cohen's Kappa | 0.8902 |
| MCC | 0.8910 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 2707.9 (0.37 ms/张) |
| End-to-End FPS | 130.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9457 | 0.9057 | 0.9143 | 0.9560 | 0.9840 | 0.9755 | 208 |
| Malignant_lesions | 0.8636 | 0.9775 | 0.9441 | 0.9021 | 0.9422 | 0.9861 | 0.9660 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9661 | 0.9360 | 0.9620 | 0.9946 | 0.9988 | 0.9976 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=20 TN=348 FN=16
- **Malignant_lesions**: TP=152 FP=9 TN=391 FN=24
- **body_of_stomach_normal**: TP=190 FP=13 TN=371 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 8 | 8 |
| Malignant_lesions | 19 | 152 | 5 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9057    0.9231    0.9143       208
     Malignant_lesions     0.9441    0.8636    0.9021       176
body_of_stomach_normal     0.9360    0.9896    0.9620       192

              accuracy                         0.9271       576
             macro avg     0.9286    0.9254    0.9261       576
          weighted avg     0.9275    0.9271    0.9265       576

```