# fast_t0_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_dce4m_seed44/fast_t0_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9474 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9477 |
| Balanced Acc | 0.9454 |
| Macro-Sensitivity | 0.9454 |
| Macro-Specificity | 0.9733 |
| Macro-Precision | 0.9519 |
| AUC-PR (Macro) | 0.9939 |
| Cohen's Kappa | 0.9215 |
| MCC | 0.9226 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 8138.4 (0.12 ms/张) |
| End-to-End FPS | 7.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9484 | 0.9144 | 0.9442 | 0.9859 | 0.9951 | 0.9920 | 208 |
| Malignant_lesions | 0.8864 | 0.9950 | 0.9873 | 0.9341 | 0.9522 | 0.9964 | 0.9916 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9766 | 0.9541 | 0.9639 | 0.9868 | 0.9990 | 0.9980 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=19 TN=349 FN=5
- **Malignant_lesions**: TP=156 FP=2 TN=398 FN=20
- **body_of_stomach_normal**: TP=187 FP=9 TN=375 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 2 | 3 |
| Malignant_lesions | 14 | 156 | 6 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9144    0.9760    0.9442       208
     Malignant_lesions     0.9873    0.8864    0.9341       176
body_of_stomach_normal     0.9541    0.9740    0.9639       192

              accuracy                         0.9479       576
             macro avg     0.9519    0.9454    0.9474       576
          weighted avg     0.9499    0.9479    0.9477       576

```