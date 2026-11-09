# starnet_s2_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_dce_last_seed43/starnet_s2_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9259 |
| Accuracy | 0.9271 |
| Macro-AUROC (OvR) | 0.9900 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9266 |
| Balanced Acc | 0.9239 |
| Macro-Sensitivity | 0.9239 |
| Macro-Specificity | 0.9628 |
| Macro-Precision | 0.9312 |
| AUC-PR (Macro) | 0.9809 |
| Cohen's Kappa | 0.8901 |
| MCC | 0.8916 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 4832.1 (0.21 ms/张) |
| End-to-End FPS | 8.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9293 | 0.8850 | 0.9217 | 0.9771 | 0.9851 | 0.9798 | 208 |
| Malignant_lesions | 0.8466 | 0.9825 | 0.9551 | 0.8976 | 0.9357 | 0.9880 | 0.9687 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9766 | 0.9536 | 0.9585 | 0.9817 | 0.9969 | 0.9943 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=26 TN=342 FN=8
- **Malignant_lesions**: TP=149 FP=7 TN=393 FN=27
- **body_of_stomach_normal**: TP=185 FP=9 TN=375 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 20 | 149 | 7 |
| body_of_stomach_normal | 6 | 1 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8850    0.9615    0.9217       208
     Malignant_lesions     0.9551    0.8466    0.8976       176
body_of_stomach_normal     0.9536    0.9635    0.9585       192

              accuracy                         0.9271       576
             macro avg     0.9312    0.9239    0.9259       576
          weighted avg     0.9293    0.9271    0.9266       576

```