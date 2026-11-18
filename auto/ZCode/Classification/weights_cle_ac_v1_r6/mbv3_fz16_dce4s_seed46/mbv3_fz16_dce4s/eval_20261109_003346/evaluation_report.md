# mbv3_fz16_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz16_dce4s_seed46/mbv3_fz16_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9658 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9967 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9720 |
| Balanced Acc | 0.9694 |
| Macro-Sensitivity | 0.9694 |
| Macro-Specificity | 0.9863 |
| Macro-Precision | 0.9624 |
| AUC-PR (Macro) | 0.9898 |
| Cohen's Kappa | 0.9555 |
| MCC | 0.9556 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 4312.3 (0.23 ms/张) |
| End-to-End FPS | 226.7 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9828 | 0.9791 | 0.9689 | 0.9661 | 0.9965 | 0.9959 | 195 |
| Malignant_lesions | 0.9494 | 0.9799 | 0.9146 | 0.9317 | 0.9884 | 0.9936 | 0.9737 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 0.9999 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=4 TN=228 FN=8
- **Malignant_lesions**: TP=75 FP=7 TN=341 FN=4
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 7 | 1 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9791    0.9590    0.9689       195
     Malignant_lesions     0.9146    0.9494    0.9317        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9719       427
             macro avg     0.9624    0.9694    0.9658       427
          weighted avg     0.9723    0.9719    0.9720       427

```