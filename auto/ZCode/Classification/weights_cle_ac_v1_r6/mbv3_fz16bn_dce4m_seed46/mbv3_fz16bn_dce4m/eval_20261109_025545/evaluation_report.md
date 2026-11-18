# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16bn_dce4m_seed46/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9527 |
| Accuracy | 0.9625 |
| Macro-AUROC (OvR) | 0.9964 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9626 |
| Balanced Acc | 0.9551 |
| Macro-Sensitivity | 0.9551 |
| Macro-Specificity | 0.9815 |
| Macro-Precision | 0.9504 |
| AUC-PR (Macro) | 0.9901 |
| Cohen's Kappa | 0.9406 |
| MCC | 0.9406 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 2331.5 (0.43 ms/张) |
| End-to-End FPS | 13.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9538 | 0.9741 | 0.9688 | 0.9612 | 0.9617 | 0.9960 | 0.9952 | 195 |
| Malignant_lesions | 0.9114 | 0.9741 | 0.8889 | 0.9000 | 0.9798 | 0.9931 | 0.9750 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=6 TN=226 FN=9
- **Malignant_lesions**: TP=72 FP=9 TN=339 FN=7
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 9 | 0 |
| Malignant_lesions | 6 | 72 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9688    0.9538    0.9612       195
     Malignant_lesions     0.8889    0.9114    0.9000        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9625       427
             macro avg     0.9504    0.9551    0.9527       427
          weighted avg     0.9628    0.9625    0.9626       427

```