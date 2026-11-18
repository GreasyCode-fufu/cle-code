# fast_t0_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/fast_t0_fz3_dce_last_seed45/fast_t0_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9443 |
| Accuracy | 0.9444 |
| Macro-AUROC (OvR) | 0.9932 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9444 |
| Balanced Acc | 0.9434 |
| Macro-Sensitivity | 0.9434 |
| Macro-Specificity | 0.9719 |
| Macro-Precision | 0.9457 |
| AUC-PR (Macro) | 0.9868 |
| Cohen's Kappa | 0.9164 |
| MCC | 0.9166 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 338.94M |
| FLOPs | 677.88M |
| Params | 2.22M |
| Pure-Inference FPS | 5880.6 (0.17 ms/张) |
| End-to-End FPS | 8.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9511 | 0.9163 | 0.9314 | 0.9695 | 0.9890 | 0.9825 | 208 |
| Malignant_lesions | 0.9091 | 0.9775 | 0.9467 | 0.9275 | 0.9607 | 0.9913 | 0.9794 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9870 | 0.9740 | 0.9740 | 0.9870 | 0.9992 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=18 TN=350 FN=11
- **Malignant_lesions**: TP=160 FP=9 TN=391 FN=16
- **body_of_stomach_normal**: TP=187 FP=5 TN=379 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 13 | 160 | 3 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9163    0.9471    0.9314       208
     Malignant_lesions     0.9467    0.9091    0.9275       176
body_of_stomach_normal     0.9740    0.9740    0.9740       192

              accuracy                         0.9444       576
             macro avg     0.9457    0.9434    0.9443       576
          weighted avg     0.9448    0.9444    0.9444       576

```