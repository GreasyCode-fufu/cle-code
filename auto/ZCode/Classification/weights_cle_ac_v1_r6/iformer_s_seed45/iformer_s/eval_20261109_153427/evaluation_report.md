# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/iformer_s_seed45/iformer_s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9176 |
| Accuracy | 0.9167 |
| Macro-AUROC (OvR) | 0.9880 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9172 |
| Balanced Acc | 0.9150 |
| Macro-Sensitivity | 0.9150 |
| Macro-Specificity | 0.9574 |
| Macro-Precision | 0.9231 |
| AUC-PR (Macro) | 0.9797 |
| Cohen's Kappa | 0.8744 |
| MCC | 0.8760 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 3163.6 (0.32 ms/张) |
| End-to-End FPS | 223.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9103 | 0.8571 | 0.9021 | 0.9710 | 0.9854 | 0.9757 | 208 |
| Malignant_lesions | 0.8920 | 0.9775 | 0.9458 | 0.9181 | 0.9537 | 0.9858 | 0.9761 | 176 |
| body_of_stomach_normal | 0.9010 | 0.9844 | 0.9665 | 0.9326 | 0.9521 | 0.9928 | 0.9872 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=33 TN=335 FN=10
- **Malignant_lesions**: TP=157 FP=9 TN=391 FN=19
- **body_of_stomach_normal**: TP=173 FP=6 TN=378 FN=19

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 7 | 3 |
| Malignant_lesions | 16 | 157 | 3 |
| body_of_stomach_normal | 17 | 2 | 173 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8571    0.9519    0.9021       208
     Malignant_lesions     0.9458    0.8920    0.9181       176
body_of_stomach_normal     0.9665    0.9010    0.9326       192

              accuracy                         0.9167       576
             macro avg     0.9231    0.9150    0.9176       576
          weighted avg     0.9207    0.9167    0.9172       576

```