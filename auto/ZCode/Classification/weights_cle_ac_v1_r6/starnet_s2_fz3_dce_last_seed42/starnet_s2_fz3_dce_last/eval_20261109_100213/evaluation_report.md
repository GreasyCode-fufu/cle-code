# starnet_s2_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_dce_last_seed42/starnet_s2_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9631 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9971 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9716 |
| Balanced Acc | 0.9589 |
| Macro-Sensitivity | 0.9589 |
| Macro-Specificity | 0.9851 |
| Macro-Precision | 0.9678 |
| AUC-PR (Macro) | 0.9902 |
| Cohen's Kappa | 0.9551 |
| MCC | 0.9553 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.23M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3961.2 (0.25 ms/张) |
| End-to-End FPS | 5.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9741 | 0.9697 | 0.9771 | 0.9869 | 0.9975 | 0.9972 | 195 |
| Malignant_lesions | 0.8987 | 0.9885 | 0.9467 | 0.9221 | 0.9773 | 0.9941 | 0.9737 | 79 |
| body_of_stomach_normal | 0.9935 | 0.9927 | 0.9870 | 0.9902 | 0.9963 | 0.9998 | 0.9997 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=6 TN=226 FN=3
- **Malignant_lesions**: TP=71 FP=4 TN=344 FN=8
- **body_of_stomach_normal**: TP=152 FP=2 TN=272 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 3 | 0 |
| Malignant_lesions | 6 | 71 | 2 |
| body_of_stomach_normal | 0 | 1 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9697    0.9846    0.9771       195
     Malignant_lesions     0.9467    0.8987    0.9221        79
body_of_stomach_normal     0.9870    0.9935    0.9902       153

              accuracy                         0.9719       427
             macro avg     0.9678    0.9589    0.9631       427
          weighted avg     0.9716    0.9719    0.9716       427

```