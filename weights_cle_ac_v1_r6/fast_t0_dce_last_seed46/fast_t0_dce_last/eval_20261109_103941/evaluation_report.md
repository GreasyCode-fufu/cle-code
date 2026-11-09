# fast_t0_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_dce_last_seed46/fast_t0_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9264 |
| Accuracy | 0.9271 |
| Macro-AUROC (OvR) | 0.9916 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9268 |
| Balanced Acc | 0.9240 |
| Macro-Sensitivity | 0.9240 |
| Macro-Specificity | 0.9627 |
| Macro-Precision | 0.9322 |
| AUC-PR (Macro) | 0.9837 |
| Cohen's Kappa | 0.8901 |
| MCC | 0.8917 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 7920.9 (0.13 ms/张) |
| End-to-End FPS | 8.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9212 | 0.8728 | 0.9128 | 0.9741 | 0.9873 | 0.9798 | 208 |
| Malignant_lesions | 0.8466 | 0.9825 | 0.9551 | 0.8976 | 0.9357 | 0.9894 | 0.9749 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9844 | 0.9688 | 0.9688 | 0.9844 | 0.9982 | 0.9965 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=29 TN=339 FN=9
- **Malignant_lesions**: TP=149 FP=7 TN=393 FN=27
- **body_of_stomach_normal**: TP=186 FP=6 TN=378 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 23 | 149 | 4 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8728    0.9567    0.9128       208
     Malignant_lesions     0.9551    0.8466    0.8976       176
body_of_stomach_normal     0.9688    0.9688    0.9688       192

              accuracy                         0.9271       576
             macro avg     0.9322    0.9240    0.9264       576
          weighted avg     0.9299    0.9271    0.9268       576

```