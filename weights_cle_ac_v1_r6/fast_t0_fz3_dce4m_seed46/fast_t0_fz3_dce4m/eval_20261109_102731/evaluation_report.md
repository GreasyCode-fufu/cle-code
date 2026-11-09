# fast_t0_fz3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_dce4m_seed46/fast_t0_fz3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9442 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9952 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9443 |
| Balanced Acc | 0.9425 |
| Macro-Sensitivity | 0.9425 |
| Macro-Specificity | 0.9716 |
| Macro-Precision | 0.9477 |
| AUC-PR (Macro) | 0.9908 |
| Cohen's Kappa | 0.9163 |
| MCC | 0.9171 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 8482.3 (0.12 ms/张) |
| End-to-End FPS | 7.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9457 | 0.9091 | 0.9346 | 0.9775 | 0.9925 | 0.9887 | 208 |
| Malignant_lesions | 0.8920 | 0.9900 | 0.9752 | 0.9318 | 0.9542 | 0.9948 | 0.9865 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9792 | 0.9590 | 0.9664 | 0.9869 | 0.9984 | 0.9970 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=20 TN=348 FN=8
- **Malignant_lesions**: TP=157 FP=4 TN=396 FN=19
- **body_of_stomach_normal**: TP=187 FP=8 TN=376 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 4 | 4 |
| Malignant_lesions | 15 | 157 | 4 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9091    0.9615    0.9346       208
     Malignant_lesions     0.9752    0.8920    0.9318       176
body_of_stomach_normal     0.9590    0.9740    0.9664       192

              accuracy                         0.9444       576
             macro avg     0.9477    0.9425    0.9442       576
          weighted avg     0.9459    0.9444    0.9443       576

```