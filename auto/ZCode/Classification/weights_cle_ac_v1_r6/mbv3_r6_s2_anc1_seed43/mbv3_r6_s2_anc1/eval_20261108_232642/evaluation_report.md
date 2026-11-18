# mbv3_r6_s2_anc1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_anc1_seed43/mbv3_r6_s2_anc1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9122 |
| Accuracy | 0.9132 |
| Macro-AUROC (OvR) | 0.9902 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9127 |
| Balanced Acc | 0.9092 |
| Macro-Sensitivity | 0.9092 |
| Macro-Specificity | 0.9554 |
| Macro-Precision | 0.9214 |
| AUC-PR (Macro) | 0.9818 |
| Cohen's Kappa | 0.8690 |
| MCC | 0.8719 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 3632.5 (0.28 ms/张) |
| End-to-End FPS | 222.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.8995 | 0.8426 | 0.8939 | 0.9707 | 0.9849 | 0.9758 | 208 |
| Malignant_lesions | 0.8068 | 0.9825 | 0.9530 | 0.8738 | 0.9204 | 0.9875 | 0.9729 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9844 | 0.9688 | 0.9688 | 0.9844 | 0.9983 | 0.9967 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=37 TN=331 FN=10
- **Malignant_lesions**: TP=142 FP=7 TN=393 FN=34
- **body_of_stomach_normal**: TP=186 FP=6 TN=378 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 6 | 4 |
| Malignant_lesions | 32 | 142 | 2 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8426    0.9519    0.8939       208
     Malignant_lesions     0.9530    0.8068    0.8738       176
body_of_stomach_normal     0.9688    0.9688    0.9688       192

              accuracy                         0.9132       576
             macro avg     0.9214    0.9092    0.9122       576
          weighted avg     0.9184    0.9132    0.9127       576

```