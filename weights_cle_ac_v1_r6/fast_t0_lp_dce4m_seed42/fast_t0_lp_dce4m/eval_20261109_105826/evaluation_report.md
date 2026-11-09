# fast_t0_lp_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_lp_dce4m_seed42/fast_t0_lp_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9375 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9930 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9374 |
| Balanced Acc | 0.9363 |
| Macro-Sensitivity | 0.9363 |
| Macro-Specificity | 0.9682 |
| Macro-Precision | 0.9397 |
| AUC-PR (Macro) | 0.9868 |
| Cohen's Kappa | 0.9059 |
| MCC | 0.9064 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 8996.4 (0.11 ms/张) |
| End-to-End FPS | 12.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9484 | 0.9116 | 0.9267 | 0.9668 | 0.9890 | 0.9832 | 208 |
| Malignant_lesions | 0.8977 | 0.9850 | 0.9634 | 0.9294 | 0.9563 | 0.9929 | 0.9827 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9714 | 0.9442 | 0.9563 | 0.9842 | 0.9971 | 0.9945 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=19 TN=349 FN=12
- **Malignant_lesions**: TP=158 FP=6 TN=394 FN=18
- **body_of_stomach_normal**: TP=186 FP=11 TN=373 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 6 | 6 |
| Malignant_lesions | 13 | 158 | 5 |
| body_of_stomach_normal | 6 | 0 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9116    0.9423    0.9267       208
     Malignant_lesions     0.9634    0.8977    0.9294       176
body_of_stomach_normal     0.9442    0.9688    0.9563       192

              accuracy                         0.9375       576
             macro avg     0.9397    0.9363    0.9375       576
          weighted avg     0.9383    0.9375    0.9374       576

```